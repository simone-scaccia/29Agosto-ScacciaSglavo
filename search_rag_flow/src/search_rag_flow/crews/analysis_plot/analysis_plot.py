from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import (
    CodeInterpreterTool,
    FileReadTool
)
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

code_interpreter = CodeInterpreterTool(unsafe_mode=True, libraries_used=['pandas', 'matplotlib', 'seaborn', 'numpy'])
file_read_csv = FileReadTool(file_path='files/AEP_hourly.csv')
file_read_docs = FileReadTool(file_path='files/docs.txt')


@CrewBase
class AnalysisPlotCrew():
    """Analysis and plotting crew for data visualization tasks.

    This crew orchestrates a three-agent workflow:
    1. File analyzer: Examines data files and documentation
    2. Code engineer: Designs analysis and visualization code
    3. Plot generator: Executes code and creates visualizations

    The crew processes CSV data files and documentation to generate
    insights and visual representations based on user queries.

    Attributes:
        agents (List[BaseAgent]): The list of instantiated agents,
            automatically created by the `@agent` decorator.
        tasks (List[Task]): The list of instantiated tasks,
            automatically created by the `@task` decorator.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def file_analyzer(self) -> Agent:
        """Create the file analyzer agent.
        
        The file analyzer agent is responsible for examining data files and
        documentation to understand the structure and content of the available
        data sources. It uses file reading tools to access CSV data and
        documentation files.
        
        Returns:
            Agent: A configured agent for file analysis with file reading tools.
        """
        return Agent(
            config=self.agents_config['file_analyzer'], # type: ignore[index]
            verbose=True,
            tools=[file_read_csv, file_read_docs]
        )

    @agent
    def code_engineer(self) -> Agent:
        """Create the code engineer agent.
        
        The code engineer agent designs and develops analysis and visualization
        code based on the file analysis results. It has access to file reading
        tools to reference data sources while creating code solutions.
        
        Returns:
            Agent: A configured agent for code engineering with file access tools.
        """
        return Agent(
            config=self.agents_config['code_engineer'], # type: ignore[index]
            verbose=True,
            tools=[file_read_csv, file_read_docs]
        )
    
    @agent
    def plot_generator(self) -> Agent:
        """Create the plot generator agent.
        
        The plot generator agent executes the analysis and visualization code
        created by the code engineer. It uses the code interpreter tool to run
        Python code safely and generate plots and visualizations. It also has
        access to file reading tools for data access.
        
        Returns:
            Agent: A configured agent for plot generation with code execution
                and file access tools.
        """
        return Agent(
            config=self.agents_config['plot_generator'], # type: ignore[index]
            tools=[code_interpreter, file_read_csv, file_read_docs],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def file_analyzer_task(self) -> Task:
        """Create the file analyzer task.
        
        This task instructs the file analyzer agent to examine the available
        data files and documentation to understand their structure and content.
        
        Returns:
            Task: A configured task for file analysis.
        """
        return Task(
            config=self.tasks_config['file_analyzer_task'], # type: ignore[index]
        )
    
    @task
    def code_engineer_task(self) -> Task:
        """Create the code engineer task.
        
        This task instructs the code engineer agent to design analysis and
        visualization code based on the file analysis results. It depends on
        the file analyzer task for context.
        
        Returns:
            Task: A configured task for code engineering with file analysis context.
        """
        return Task(
            config=self.tasks_config['code_engineer_task'], # type: ignore[index]
            context=[self.file_analyzer_task()]
        )
    
    @task
    def plot_generator_task(self) -> Task:
        """Create the plot generator task.
        
        This task instructs the plot generator agent to execute the analysis
        and visualization code created by the code engineer. It depends on the
        code engineer task for the code to execute.
        
        Returns:
            Task: A configured task for plot generation with code engineering context.
        """
        return Task(
            config=self.tasks_config['plot_generator_task'], # type: ignore[index]
            context=[self.code_engineer_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Create the assembled AnalysisPlot crew.
        
        The crew executes tasks sequentially: file analysis -> code engineering
        -> plot generation. Each task builds upon the previous one to create
        a complete data analysis and visualization workflow.
        
        Returns:
            Crew: The fully assembled crew with agents and tasks configured
                for sequential processing.
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
