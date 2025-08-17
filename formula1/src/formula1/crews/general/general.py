import requests
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai.tools import tool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

llm = LLM(
    model="ollama/qwen3:0.6b",
    base_url="http://localhost:11434",
    temperature=1,
    config={
        "max_tokens": 128,
        "top_k": 5,
    }
)

@tool("Wikipedia Search Tool")
def myWikipediaSearch(question: str) -> str:
    """Searches Wikipedia for the given query and returns the first paragraph."""
    
    url="https://en.wikipedia.org/w/api.php"
    params={
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": question
    }

    response = requests.get(url, params=params)
    results = response.json().get('query', {}).get('search', [])

    if not results:
        return "No results found."

    title = results[0]['title']
    
    newParams={
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title
    }

    newResponse = requests.get(url, params=newParams)
    page = next(iter(newResponse.json().get('query', {}).get('pages', {}).values()), {})
    extract = page.get('extract', 'No extract found.')

    return f"Title: {title}\nExtract: {extract}"

@CrewBase
class General():
    """General Crew"""

    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"
    
    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            tools=[myWikipediaSearch],
            llm=llm,
        )

#    @agent
#    def summarizer(self) -> Agent:
#        return Agent(
#            config=self.agents_config['summarizer'], # type: ignore[index]
#            verbose=True,
#            llm=llm,
#        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research(self) -> Task:
        return Task(
            config=self.tasks_config['research'], # type: ignore[index]
        )

#   @task
#    def summarize(self) -> Task:
#        return Task(
#            config=self.tasks_config['summarize'], # type: ignore[index]
            # context=[research],
            # output_file='report.md'
#       )

    @crew
    def crew(self) -> Crew:
        """Creates the General Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            llm=llm,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
