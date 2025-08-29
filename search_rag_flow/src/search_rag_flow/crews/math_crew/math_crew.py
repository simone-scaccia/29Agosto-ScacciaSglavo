"""Math crew definitions and configuration.

This module declares the `MathCrew` class using CrewAI decorators to
assemble agents and tasks involved in generating and evaluating
mathematical functions. It exposes factory methods for two agents,
two tasks, and a `Crew` instance that wires them together in a
sequential process.

The concrete agent and task configurations are expected to be provided
via YAML files loaded by CrewAI (referenced through
`self.agents_config` and `self.tasks_config`).
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from search_rag_flow.tools.math_tool import validate_math_expression
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class MathCrew():
    """Orchestrates agents and tasks for a math-focused crew.

    This crew is responsible for building a mathematical function based on
    requirements and then evaluating it. The underlying configuration for
    agents and tasks is read from YAML files referenced by CrewAI.

    Attributes:
        agents (List[BaseAgent]):
            The list of instantiated agents, automatically created by the
            `@agent` decorator.
        tasks (List[Task]):
            The list of instantiated tasks, automatically created by the
            `@task` decorator.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def function_builder(self) -> Agent:
        """Create the function builder agent.

        The function builder agent is configured from the
        `function_builder` section in the agents configuration and is
        equipped with the `validate_math_expression` tool to ensure the
        mathematical function syntax is valid.

        Returns:
            Agent: A configured agent responsible for building a
            mathematical function.
        """
        return Agent(
            config=self.agents_config['function_builder'], # type: ignore[index]
            verbose=True,
            tools=[validate_math_expression] # type: ignore[index]
        )

    @agent
    def function_evaluator(self) -> Agent:
        """Create the function evaluator agent.

        The function evaluator agent is configured from the
        `function_evaluator` section in the agents configuration and is
        used to assess or verify the function produced by the builder.

        Returns:
            Agent: A configured agent responsible for evaluating the
            generated function.
        """
        return Agent(
            config=self.agents_config['function_evaluator'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def build_function_task(self) -> Task:
        """Create the task that instructs building the function.

        The task configuration is loaded from the
        `build_function_task` section in the tasks configuration.

        Returns:
            Task: The task responsible for guiding the creation of a
            mathematical function.
        """
        return Task(
            config=self.tasks_config['build_function_task'], # type: ignore[index]
        )

    @task
    def evaluate_function_task(self) -> Task:
        """Create the task that evaluates the built function.

        The task configuration is loaded from the
        `evaluate_function_task` section in the tasks configuration. The
        output of this task is written to `math_result.md`.

        Returns:
            Task: The task responsible for evaluating the mathematical
            function and saving the results.
        """
        return Task(
            config=self.tasks_config['evaluate_function_task'], # type: ignore[index]
            output_file='math_result.md'
        )

    @crew
    def crew(self) -> Crew:
        """Create and wire the `MathCrew` crew.

        The crew executes tasks sequentially using the agents defined
        above. Verbose mode is enabled for detailed logging during
        execution.

        Returns:
            Crew: The configured crew with agents and tasks assembled in a
            sequential process.
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
