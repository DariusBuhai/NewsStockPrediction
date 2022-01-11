from modules.agent import Agent
import os

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    agent = Agent("PFE")

    agent.train(steps=10)
    agent.evaluate()
    agent.show()