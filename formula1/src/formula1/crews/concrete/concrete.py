import csv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import CSVSearchTool
# from langchain.embeddings import HuggingFaceEmbeddings
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

llm = LLM(
    model="ollama/qwen3:0.6b",
    base_url=f"http://localhost:11434",
    temperature=1,
    config={
        "max_tokens": 128,
        "top_k": 5,
    }
)

embedchain_config = {
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "all-minilm:22m",
        }
    }
}

# embedchain_config=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

drivers=CSVSearchTool(csv='/home/extensa/Projects/Formula1/Formula1/formula1/knowledge/drivers.csv', config=embedchain_config)
circuits=CSVSearchTool(csv='/home/extensa/Projects/Formula1/Formula1/formula1/knowledge/circuits.csv', config=embedchain_config)
constructors=CSVSearchTool(csv='/home/extensa/Projects/Formula1/Formula1/formula1/knowledge/constructors.csv', config=embedchain_config)

@CrewBase
class Concrete():
    """Concrete Crew"""

    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def concrete(self) -> Agent:
        return Agent(
            config=self.agents_config['concrete'], # type: ignore[index]
            verbose=True,
            tools=[drivers, circuits, constructors],
            llm=llm,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def qa(self) -> Task:
        return Task(
            config=self.tasks_config['qa'], # type: ignore[index]
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
            # embedder=embedder_config,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
