import asyncio
import logging
import json
import subprocess
import sys

from dataclasses import dataclass
from devtools import pprint

import colorlog
import httpx
from httpx import AsyncClient
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.vertexai import VertexAIModel

import logfire

logfire.configure(send_to_logfire=False, console=logfire.ConsoleOptions(verbose=True))

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

pre_path = "/home/thoraxe/bin/"

log_format = "%(log_color)s%(asctime)s [%(levelname)s] %(reset)s%(purple)s[%(name)s] %(reset)s%(blue)s%(message)s"
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(log_format))
logging.basicConfig(level=logging.DEBUG, handlers=[handler])

logger = logging.getLogger(__name__)


@dataclass
class CLIDependencies:
    token: str


class CLIResult(BaseModel):
    output: str = Field(description="The output from the CLI command")


agent_extras = """
In general:
* when it can provide extra information, first run as many tools as you need to gather more information, then respond. 
* if possible, do so repeatedly with different tool calls each time to gather more information.
* do not stop investigating until you are at the final root cause you are able to find. 
* use the "five whys" methodology to find the root cause.
* for example, if you found a problem in microservice A that is due to an error in microservice B, look at microservice B too and find the error in that.
* if you cannot find the resource/application that the user referred to, assume they made a typo or included/excluded characters like - and.
* in this case, try to find substrings or search for the correct spellings
* if you are unable to investigate something properly because you do not have access to the right data, explicitly tell the user that you are missing an integration to access XYZ which you would need to investigate. you should specifically use the templated phrase "I don't have access to <details>. Please add a Holmes integration for <XYZ> so that I can investigate this."
* always provide detailed information like exact resource names, versions, labels, etc
* even if you found the root cause, keep investigating to find other possible root causes and to gather data for the answer like exact names
* if a runbook url is present as well as tool that can fetch it, you MUST fetch the runbook before beginning your investigation.
* if you don't know, say that the analysis was inconclusive.
* if there are multiple possible causes list them in a numbered list.
* there will often be errors in the data that are not relevant or that do not have an impact - ignore them in your conclusion if you were not able to tie them to an actual error.
* run as many kubectl commands as you need to gather more information, then respond.
* if possible, do so repeatedly on different Kubernetes objects.
* for example, for deployments first run kubectl on the deployment then a replicaset inside it, then a pod inside that.
* when investigating a pod that crashed or application errors, always run kubectl_describe and fetch logs with both kubectl_previous_logs and kubectl_logs so that you see current logs and any logs from before a crash.
* do not give an answer like "The pod is pending" as that doesn't state why the pod is pending and how to fix it.
* do not give an answer like "Pod's node affinity/selector doesn't match any available nodes" because that doesn't include data on WHICH label doesn't match
* if investigating an issue on many pods, there is no need to check more than 3 individual pods in the same deployment. pick up to a representative 3 from each deployment if relevant
* if you find errors and warning in a pods logs and you believe they indicate a real issue. consider the pod as not healthy. 
* if the user says something isn't working, ALWAYS:
** use kubectl_describe on the owner workload + individual pods and look for any transient issues they might have been referring to
** check the application aspects with kubectl_logs + kubectl_previous_logs and other relevant tools
** look for misconfigured ingresses/services etc

Style guide:
* Be painfully concise.
* Leave out "the" and filler words when possible.
* Be terse but not at the expense of leaving out important data like the root cause and how to fix.
"""


# the routing agent chooses to use either the knowledge agent or the retrieval
# agent via their tools
routing_agent = Agent(
    "openai:gpt-4o",
    name="routing_agent",
    system_prompt="""You are a Kubernetes and OpenShift assistant. You should
    only answer questions related to OpenShift and Kubernetes. You can retrieve
    information from Kubernetes and OpenShift environments using your tools. You
    also have a tool that can answer general knowledge questions. Always use the
    original unmodified user question when calling a tool.""",
)


@routing_agent.tool
async def knowledge_tool(ctx: RunContext[str], original_query: str) -> str:
    """A tool for answering general OpenShift and Kubernetes knowledge
    questions. Use for obtaining how-to, documentation, and similar answers.

    Args:
        original_query: the question to get an answer for
    """

    logger.debug(ctx)
    r = await knowledge_agent.run(original_query, usage=ctx.usage)

    # pprint(r.all_messages)
    return r.data


@routing_agent.tool
async def retrieval_tool(ctx: RunContext[str], original_query: str) -> str:
    """A tool for retrieving information from a running OpenShift cluster.

    Args:
        original_query: the question to get an answer for
    """

    logger.debug(ctx)
    r = await retrieval_agent.run(original_query, usage=ctx.usage)

    return r.data


knowledge_agent = Agent(
    "openai:gpt-4o",
    name="knowledge_agent",
    system_prompt="""You are a Kubernetes and OpenShift assistant. You should
only answer questions related to OpenShift and Kubernetes. You are supposed
to answer general knowledge, how-to, documentation, and other similar
questions about OpenShift and Kubernetes. Prefer OpenShift-specific answers
and try to avoid use of kubectl and other generic Kubernetes knowledge. Assume
the user is always asking questions about Openshift.
""",
)

retrieval_agent = Agent(
    "openai:gpt-4o",
    name="retrieval_agent",
    system_prompt="""You are a Kubernetes and OpenShift assistant. You should
only answer questions related to OpenShift and Kubernetes. You can retrieve
information from Kubernetes and OpenShift environments using your tools. Assume
the user is always asking questions about OpenShift. Use the oc command line
tool and do not use the kubectl tool when describing solutions.

""",
)


@retrieval_agent.tool
def get_pod_list(ctx: RunContext[str], namespace: str) -> str:
    """Get the list of pods in a specific namespace.
    Args:
        namespace: the namespace to get the pod list from
    """

    output = subprocess.run(
        [pre_path + "oc", "get", "pods", "-n", namespace, "-o", "name"],
        capture_output=True,
        timeout=2,
    )
    logger.debug(output.stdout)
    return output.stdout


@retrieval_agent.tool
def get_namespaces(ctx: RunContext[str]) -> str:
    """Get the list of namespaces in the cluster."""
    output = subprocess.run(
        [pre_path + "oc", "get", "namespaces"], capture_output=True, timeout=2
    )
    logger.debug(output.stdout)
    return output.stdout


@retrieval_agent.tool
async def get_nonrunning_pods(ctx: RunContext[str]) -> str:
    """Get the list of pods in the cluster that are not currently running.  A
    pod that is not running is not necessarily broken. You should check the pod
    status to determine why it is not running.
    """
    output = subprocess.run(
        [
            pre_path + "oc",
            "get",
            "pods",
            "-A",
            "--field-selector",
            "status.phase!=Running",
            "-o",
            "custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name",
        ],
        capture_output=True,
        timeout=2,
    )
    logger.debug(output.stdout)
    return output.stdout


@retrieval_agent.tool
async def get_pod_details(ctx: RunContext[str], namespace: str, pod: str) -> str:
    """Returns the entire YAML for a specific pod in a specific namespace.

    Args:
        pod: the name of the pod to check
        namespace: the namespace where the pod exists
    """

    output = subprocess.run(
        [pre_path + "oc", "get", "pod", "-n", namespace, pod, "-o", "yaml"],
        capture_output=True,
        timeout=2,
    )
    logger.debug(output.stdout)
    return output.stdout


@retrieval_agent.tool
async def get_pod_status(ctx: RunContext[str], namespace: str, pod: str) -> str:
    logger.debug(f"get pod status: {pod}/{namespace}")
    """Returns only the status object for a specific pod in a specific namespace.

    Args:
        pod: the name of the pod to check
        namespace: the namespace where the pod exists
    """
    cmd = [
            pre_path + "oc",
            "get",
            "pod",
            "-n",
            namespace,
            pod,
            "-o",
            "jsonpsth=\{.status\}",
        ]

    logger.debug(cmd)
    output = subprocess.run(
        cmd,
        capture_output=True,
        timeout=2,
    )
    logger.debug(output.stdout)
    logger.debug(output.stderr)
    return output.stdout


@routing_agent.system_prompt
@retrieval_agent.system_prompt
def add_extras() -> str:
    return agent_extras


result = routing_agent.run_sync(sys.argv[1])
print(result.data)

pprint(result.usage())