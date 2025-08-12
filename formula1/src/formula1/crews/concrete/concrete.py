import warnings, re, csv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, message=r".*\\&.*")

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import CSVSearchTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

llm = LLM(
    model="gemma3:1b",
    temperature=0.3,
    config={
        "max_tokens": 256,
        "top_k": 10,
    }
)

embedchain_config = {
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
        }
    }
}

drivers=CSVSearchTool(csv='knowledge/drivers.csv', config=embedchain_config)
circuits=CSVSearchTool(csv='knowledge/circuits.csv', config=embedchain_config)
constructors=CSVSearchTool(csv='knowledge/constructors.csv', config=embedchain_config)

@CrewBase
class Concrete():
    """Concrete Crew"""

    agents="config/agents.yaml"
    tasks="config/tasks.yaml"

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def concrete(self) -> Agent:
        return Agent(
            config=self.agents['concrete'], # type: ignore[index]
            verbose=True,
            tools=[drivers, circuits, constructors],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def qa(self) -> Task:
        return Task(
            config=self.tasks['qa'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Concrete Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            llm=llm,
            embedder=embedder_config,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
