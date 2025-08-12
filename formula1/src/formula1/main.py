#!/usr/bin/env python
from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from formula1.crews.classifier.classifier import Classifier
from formula1.crews.concrete.concrete import Concrete
from formula1.crews.general.general import General
from formula1.crews.others.others import Others

class UserInput(BaseModel):
    input: str = ""

class MainFlow(Flow[UserInput]):

    @start()
    def getUserInput(self):
        print("What are you looking for?")
        self.state.input = input("You:")

    @listen(getUserInput)
    def startFlow(self):
        inputs={
            "user_message": self.input,
        }

        response=""

        try:
            raw=Classifier.crew().kickoff(inputs=inputs)
            decision=raw['classification']
        except Exception as e:
            raise Exception(f"An error occurred while running the classifier crew: {e}")

        if decision=='general':
            try:
                response=General.crew().kickoff(intputs=inputs)
            except Exception as e:
                raise Exception(f"An error occurred while running the general crew: {e}")
        elif decision=='concrete':
            try:
                response=Concrete.crew().kickoff(intputs=inputs)
            except Exception as e:
                raise Exception(f"An error occurred while running the concrete crew: {e}")
        elif decision=='others':
            try:
                response=Others.crew().kickoff(intputs=inputs)
            except Exception as e:
                raise Exception(f"An error occurred while running the others crew: {e}")
        else:
            print("Houston, we've had a problem.")

        print response

def kickoff():
    flow = MainFlow(UserInput())
    flow.kickoff()

def plot():
    flow = MainFlow(UserInput())
    flow.plot()

if __name__ == "__main__":
    kickoff()
