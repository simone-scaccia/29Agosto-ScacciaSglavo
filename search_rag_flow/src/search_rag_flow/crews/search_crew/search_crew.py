"""Search crew definitions and configuration.

This module defines the `SearchCrew` which orchestrates a simple
two-agent workflow: a researcher that gathers information using a
custom DuckDuckGo-based tool, and a summarizer that condenses the
findings. It exposes tasks to perform the research and produce a
summary, and bundles them into a sequential `Crew`.

The implementation relies on `crewai` decorators to declaratively
define agents, tasks, and the crew composition.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from search_rag_flow.tools.research_ddg_tool import research_ddg_tool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class SearchCrew():
    """Crew that performs a research-and-summarize workflow.

    This crew defines two agents:

    - A researcher that uses a DuckDuckGo research tool to find relevant
      information.
    - A summarizer that synthesizes the collected information into a
      concise report.

    The crew runs its tasks sequentially by default.
    """

    agents: List[BaseAgent]
    tasks: List[Task]
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        """Create and configure the researcher agent.

        The researcher leverages the `research_ddg_tool` to perform
        web searches and collect information relevant to the research
        task.

        Returns:
            Agent: A configured `Agent` instance representing the
            researcher.
        """
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            tools=[research_ddg_tool]
        )

    @agent
    def summarizer(self) -> Agent:
        """Create and configure the summarizer agent.

        The summarizer is responsible for reading the research output
        and producing a concise, clear summary.

        Returns:
            Agent: A configured `Agent` instance representing the
            summarizer.
        """
        return Agent(
            config=self.agents_config['summarizer'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        """Define the research task executed by the researcher agent.

        This task instructs the researcher to gather information based
        on the provided configuration.

        Returns:
            Task: A `Task` instance configured for research.
        """
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def summary_task(self) -> Task:
        """Define the summary task executed by the summarizer agent.

        The task consolidates research findings and writes the result to
        an output file.

        Returns:
            Task: A `Task` instance configured to produce a summary and
            write it to ``report.md``.
        """
        return Task(
            config=self.tasks_config['summary_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Create the `Crew` that runs the research and summary tasks.

        Returns:
            Crew: A sequential crew composed of the researcher and
            summarizer agents executing the research and summary tasks.
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
