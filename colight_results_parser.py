from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class TrainResults:
    """Class for keeping track of train results."""
    episode: int
    q_loss: float
    rewards: float
    travel_time: float

    def __init__(self, episode: int, q_loss: float, rewards: float, travel_time: int):
        self.episode = episode
        self.q_loss = q_loss
        self.rewards = rewards
        self.travel_time = travel_time


@dataclass
class TestResults:
    """Class for keeping track of testing results."""
    episode: int
    travel_time: float
    rewards: float

    def __init__(self, episode: int, travel_time: float, rewards: float):
        self.episode = episode
        self.rewards = rewards
        self.travel_time = travel_time


def parse_results(file_path: str):
    training_results = []
    testing_results = []

    with open(file_path) as fp:
        entire_file = fp.read()

    entire_file = entire_file.split('\n')

    for row in entire_file:
        split_row = row.split()
        if "q_loss" in row:
            # Handle first train case.
            colon_split = {x.split(':')[0]: x.split(':')[1].split(',')[0] for x in split_row}
            training_results.append(TrainResults(None, float(colon_split["q_loss"]), float(colon_split["rewards"]), None))
        if "average travel time" in row:
            # Handle second train case.
            split_row = row.split(", ")
            colon_split = {x.split(':')[0]: x.split(':')[1].split("/")[0] for x in split_row}
            training_results[-1].episode = int(colon_split["episode"])
            training_results[-1].travel_time = float(colon_split["average travel time"])
        elif "Test" in row:
            # Handle test case.
            split_row = row.split(", ")
            colon_split = {x.split(':')[0]: x.split(':')[1].split("/")[0] for x in split_row}
            testing_results.append(TestResults(int(colon_split["Test step"]), float(colon_split["travel time "]),
                                               float(colon_split["rewards"])))

    return training_results, testing_results


def plot_results(training_results: list, testing_results: list, model_name: str = ""):
    # TODO: Also make plots comparing the two models.

    # Travel times plot
    training_travel_times = [x.travel_time for x in training_results]
    testing_travel_times = [x.travel_time for x in testing_results]

    plt.title(f"Travel times during training, model={model_name}")
    plt.plot(training_travel_times, label="Training data")
    plt.plot(testing_travel_times, label="Testing data")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Average travel time")
    plt.savefig(fname=f"./colight_results/plots/{model_name}_travel_times.png")
    plt.show()

    # Rewards plot
    train_rewards = [x.rewards for x in training_results]
    test_rewards = [x.rewards for x in testing_results]

    plt.title(f"Rewards during training, model={model_name}")
    plt.plot(train_rewards, label="Training data")
    plt.plot(test_rewards, label="Testing data")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig(fname=f"./colight_results/plots/{model_name}_rewards.png")
    plt.show()

    # Q_loss plot
    train_q_loss = [x.q_loss for x in training_results]

    plt.title(f"Q loss during training, model={model_name}")
    plt.plot(train_q_loss, label="Training data")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Q loss")
    plt.savefig(fname=f"./colight_results/plots/{model_name}_q_loss.png")
    plt.show()


if __name__ == "__main__":
    # Enter txt file.
    results_txt = "colight_results/hidden_no_autoencoder1.txt"

    # Parse the input
    training_results_list, testing_results_list = parse_results(results_txt)

    # Plot results
    plot_results(training_results_list, testing_results_list, results_txt.split("/")[-1].split(".txt")[0])
