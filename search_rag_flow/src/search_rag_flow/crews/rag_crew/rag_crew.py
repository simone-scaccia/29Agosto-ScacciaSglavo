"""RAG crew configuration and assembly for the search_rag_flow project.

This module defines the `RagCrew`, a set of agents and tasks orchestrated with
`crewai` to perform a Retrieval-Augmented Generation (RAG) workflow. The crew
is built using YAML-based configurations for agents and tasks, and it wires in
project-specific tools where needed.

The crew runs sequentially by default:
1) Refine the user query
2) Search documents with a retrieval tool
3) Validate and generate the final response

Note:
- Agent and task configurations are expected to be present in the YAML files
  referenced by `self.agents_config` and `self.tasks_config`, which are provided
  by the `@CrewBase` decorator integration.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from search_rag_flow.tools.rag_tool import rag_tool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class RagCrew():
    """RAG crew that refines queries, searches documents, and validates responses.

    This crew wires together three agents and three tasks to execute a
    sequential Retrieval-Augmented Generation flow. Agent and task settings are
    sourced from YAML configuration files made available through the
    `@CrewBase` decorator.

    Attributes:
        agents (List[BaseAgent]):
            The list of instantiated agents, automatically created by
            the `@agent`-decorated factory methods.
        tasks (List[Task]):
            The list of instantiated tasks, automatically created by
            the `@task`-decorated factory methods.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def query_refiner(self) -> Agent:
        """Create the query refiner agent.

        The query refiner improves or restructures the initial user prompt to
        optimize retrieval and downstream reasoning.

        Returns:
            Agent: Configured query refiner agent.
        """
        return Agent(
            config=self.agents_config['query_refiner'], # type: ignore[index]
            verbose=True
        )

    @agent
    def document_searcher(self) -> Agent:
        """Create the document searcher agent.

        This agent uses retrieval tooling to locate and summarize relevant
        documents. It is equipped with the project-specific `rag_tool`.

        Returns:
            Agent: Configured document searcher agent with tools attached.
        """
        return Agent(
            config=self.agents_config['document_searcher'], # type: ignore[index]
            verbose=True,
            tools=[rag_tool] # type: ignore[index]
        )

    @agent
    def response_validator(self) -> Agent:
        """Create the response validator agent.

        The validator reviews retrieved context and candidate answers to ensure
        correctness, grounding, and clarity before producing the final output.

        Returns:
            Agent: Configured response validator agent.
        """
        return Agent(
            config=self.agents_config['response_validator'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def refine_query_task(self) -> Task:
        """Define the task that refines the user query.

        Returns:
            Task: Task configured to refine or rewrite the initial query.
        """
        return Task(
            config=self.tasks_config['refine_query_task'], # type: ignore[index]
        )

    @task
    def search_documents_task(self) -> Task:
        """Define the task that searches for relevant documents.

        Returns:
            Task: Task configured to retrieve and synthesize context documents.
        """
        return Task(
            config=self.tasks_config['search_documents_task'], # type: ignore[index]
        )

    @task
    def validate_and_respond_task(self) -> Task:
        """Define the task that validates context and produces the final response.

        The task writes its result to ``rag_response.md`` in the project root.

        Returns:
            Task: Task configured to validate the answer and generate output.
        """
        return Task(
            config=self.tasks_config['validate_and_respond_task'], # type: ignore[index]
            output_file='rag_response.md'
        )

    @crew
    def crew(self) -> Crew:
        """Create the assembled RAG crew.

        The crew runs its tasks in a sequential process by default.

        Returns:
            Crew: The fully assembled crew with agents and tasks.
        """
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
