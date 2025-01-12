import asyncio
import logging
import json
import os
import subprocess
import sys
import time

from dataclasses import dataclass
from devtools import pprint

import colorlog
import httpx
from httpx import AsyncClient
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.vertexai import VertexAIModel

from shutil import which

log_format = "%(log_color)s%(asctime)s [%(levelname)s] %(reset)s%(purple)s[%(name)s] %(reset)s%(blue)s%(message)s"
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(log_format))

#logfilename = f"{int(time.time())}.log"
#filehandler = logging.FileHandler(filename=logfilename, mode="a")
logging.basicConfig(level=logging.INFO, handlers=[handler])

logger = logging.getLogger(__name__)

# only for the spans
#file_logger = logging.getLogger(__name__)
#file_logger.handlers=filehandler

from opentelemetry.sdk.trace.export import ConsoleSpanExporter, ReadableSpan, SimpleSpanProcessor
import logfire

traces_endpoint = 'http://127.0.0.1:4318/v1/traces'
os.environ['OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'] = traces_endpoint

logfire.configure(send_to_logfire=False, 
                  console=logfire.ConsoleOptions(verbose=True, min_log_level="debug"))

def formatter(span: ReadableSpan):
    the_span = span.to_json(indent=None) + '\n'
    return the_span

#logfire.configure(send_to_logfire=False, 
#                  console=logfire.ConsoleOptions(verbose=True, min_log_level="debug"),
#                  additional_span_processors=[SimpleSpanProcessor(ConsoleSpanExporter(formatter=formatter))])

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

agent_extras = """
In general:
* come up with a plan for what you want to do and then execute the plan step by step
* always find a list of objects of a particular type first before investigating further
* when it can provide extra information, first run as many tools as you need to gather more information, then respond. 
* if possible, do so repeatedly with different tool calls each time to gather more information.
* do not stop investigating until you are at the final root cause you are able to find. 
* use the "five whys" methodology to find the root cause.
* make sure you are using information you gathered in previous steps in next steps. do not assume cluster resources exist from prior knowledge.

Style guide:
* Be painfully concise.
* Do not summarize specific details in the output of tools - use the details from the tools when possible
* Leave out "the" and filler words when possible.
* Be terse but not at the expense of leaving out important data like the root cause and how to fix.
"""


# the routing agent chooses to use either the knowledge agent or the retrieval
# agent via their tools
routing_agent = Agent(
    "openai:gpt-4o",
    name="routing_agent",
    system_prompt="""You are a Kubernetes and OpenShift assistant. You should
    only answer questions related to OpenShift, Kubernetes, and metrics from Prometheus. You have multiple
    agents on your team.
    
    You can retrieve information from Kubernetes and OpenShift environments
    using the retrieval agent. 
    
    You can also get a PromQL query from the metrics agent, to answer any system-level questions, that cannot only be
    answered by the retrieval agent or Kubernetes/OpenShift knowledge.
    
    You also have an agent that can answer general knowledge questions. 
    
    Always use the original unmodified user question when calling a tool or
    talking to your agents. 
    
    Do not summarize the output from your tools or agents.
    """,
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


@routing_agent.tool
async def metrics_tool(ctx: RunContext[str], original_query: str) -> str:
    """A tool for generating PromQL queries for a running OpenShift cluster's Prometheus instance.
    Use this tool to answer questions about system-level metrics or things that cannot be only answered by the retrieval agent.

    Args:
        original_query: the question to get an answer for
    """

    logger.debug(ctx)
    r = await metrics_agent.run(original_query, usage=ctx.usage)

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
only answer questions related to OpenShift and Kubernetes. You have access to
the OpenShift environment and you can retrieve information from Kubernetes and
OpenShift environments using your tools. Assume the user is always asking
questions about OpenShift and try to determine how to retrieve information in
that scenario.
""",
)

metrics_agent = Agent(
    "openai:gpt-4o",
    name="metrics_agent",
    system_prompt="""You are a Prometheus assistant. You should
only answer questions related to metrics stored in a cluster's Prometheus instance.
You can use the tool get_existing_metrics to get a list of metrics available in the Prometheus TSDB. 

Using those construct a PromQL query that will answer the question, in the same way an SRE would, assuming system knowledge.
Ensure that,
- The PromQL query is valid PromQL and will not cause errors and can actually run.
- The PromQL query is URL encodable.
- The PromQL query takes into account the upstream and open source best practices and norms for Prometheus and PromQL.
- The PromQL query make reasonable assumptions from the query and the metrics provided as well as their nomenclature.
- Your final PromQL query has balanced brackets and balanced double quotes (when dealing with label selectors)

Also provide an SRE-like explanation of the query and the metrics it is based on .
""",
)

def is_command_available(command: str) -> bool:
    """Check if a shell command is available in the system PATH.
    
    Args:
        command: Name of the command to check
        
    Returns:
        bool: True if command exists, False otherwise
    """
    return which(command) is not None

@retrieval_agent.tool
def get_namespaces(ctx: RunContext[str]) -> str:
    """Get the list of namespaces in the cluster."""
    if not is_command_available("oc"):
        return "Error: 'oc' command not found in PATH"
    
    output = subprocess.run(
        ["oc", "get", "namespaces"], capture_output=True, timeout=2
    )
    logger.debug(output.stdout)
    return output.stdout

@retrieval_agent.tool
def get_object_cluster_wide_list(ctx: RunContext[str], kind: str) -> str:
    """
    Fetch a list of all instances of a specific type of kubernetes/openshift 
    object in the cluster.

    Args:
        kind: the kubernetes/openshift objects to get

    Returns:
        str: the list of objects in the cluster
    """
    if not is_command_available("oc"):
        return "Error: 'oc' command not found in PATH"

    logger.info(f"provided kind: {kind}")
    output = subprocess.run(
        ["oc", "get", kind, "-A", "-o", "name"],
        capture_output=True,
        timeout=2,
    )
    return output.stdout

@retrieval_agent.tool
def get_object_namespace_list(ctx: RunContext[str], kind: str, namespace: str) -> str:
    """
    Fetch a list of all instance of a specific type of kubernetes/openshift
    object in a specific namespace.

    Args:
        kind: the kubernetes/openshift objects to get
        namespace: the namespace containing the objects

    Returns:
        str: the list of objects in the namespace
    """    
    if not is_command_available("oc"):
        return "Error: 'oc' command not found in PATH"

    output = subprocess.run(
        ["oc", "get", kind, "-n", namespace, "-o", "name"],
        capture_output=True,
        timeout=2,
    )
    return output.stdout

@retrieval_agent.tool
async def get_nonrunning_pods(ctx: RunContext[str]) -> str:
    """Get the list of pods in the cluster that are not currently running.  A
    pod that is not running is not necessarily broken. You should check the pod
    status to determine why it is not running.
    """
    if not is_command_available("oc"):
        return "Error: 'oc' command not found in PATH"

    output = subprocess.run(
        [
            "oc",
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
def get_object_details(ctx: RunContext[str], namespace: str, kind: str, name: str) -> str:
    """
    Fetch the details for a specific object in the cluster.

    Args:
        namespace: the namespace where the object is
        kind: the kind of the object
        name: the name of the object

    Returns:
        str: the YAML text of the object
    """
    if not is_command_available("oc"):
        return "Error: 'oc' command not found in PATH"

    output = subprocess.run(
        ["oc", "get", kind, "-n", namespace, name, "-o", "yaml"],
        capture_output=True,
        timeout=2,
    )
    return output.stdout

@retrieval_agent.tool
def get_pod_list(ctx: RunContext[str], namespace: str) -> str:
    """Get the list of pods in a specific namespace.
    Args:
        namespace: the namespace to get the pod list from
    """
    if not is_command_available("oc"):
        return "Error: 'oc' command not found in PATH"

    output = subprocess.run(
        ["oc", "get", "pods", "-n", namespace, "-o", "name"],
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

    Returns:
        string: the YAML status block for the object
    """
    if not is_command_available("oc") or not is_command_available("yq"):
        return "Error: Required commands ('oc' and/or 'yq') not found in PATH"

    output = subprocess.run(f"oc get pod -n {namespace} {pod} -o jsonpath='{{.status}}' | yq -p json -o yaml",
        shell=True,
        capture_output=True,
        timeout=2,
    )
    return output.stdout

@retrieval_agent.tool
async def get_object_health(ctx: RunContext[str], kind: str, name: str, **kwargs) -> str:
    """A simple tool to describe the health of an object in the cluster. Must be
    used on individual objects one at a time. Does not accept 'all' as a name.
    For example, if you want to look at all nodes, you must run this tool one at
    a time against each individual node.

    Args:
      (optional) namespace: the namespace where the object is. not used for cluster-scoped objects
      kind: the type of object
      name: the name of the object

    Returns:
      str: text describing the health of the object
    """
    if not is_command_available("kube-health"):
        return "Error: 'kube-health' command not found in PATH"

    namespace = kwargs.get('namespace', None)

    logger.info(f"get_object_health: {namespace} {kind}/{name}")
    if namespace is None:
        output = subprocess.run(f"kube-health -H {kind}/{name}",
                                shell=True,
                                capture_output=True,
                                timeout=2
                                )
    else:
        output = subprocess.run(f"kube-health -n {namespace} -H {kind}/{name}",
                                shell=True,
                                capture_output=True,
                                timeout=2
                                )
    
    nlines = len(output.stdout.splitlines())

    if nlines < 2:
        return "Error: The object you are looking for does not exist"
    else:
        return output.stdout
    
@metrics_agent.tool
async def get_existing_metrics(ctx: RunContext[str], original_query: str) -> list[str] | None:
    """
    Get a list of metrics that are relevant to the query.

    Args:
        original_query: the question to get an relevant metrics for, from the in-cluster Prometheus instance

    Returns:
        list[str] | None: a list of metrics with all their label name and values, that are relevant to the query,
                         or None if an error occurs
    """
    url = "http://localhost:8081/similar_metrics"
    params = {
        "prompt": original_query,
        "topk": 10
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            print(data[0]["series"])
            return data[0]["series"]
    except httpx.RequestError as e:
        print(f"An error occurred while requesting: {e}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    

@routing_agent.system_prompt
@retrieval_agent.system_prompt
def add_extras() -> str:
    return agent_extras

result = routing_agent.run_sync(sys.argv[1])
print(result.data)

pprint(result.usage())