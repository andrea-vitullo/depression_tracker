import matplotlib.pyplot as plt
import numpy as np


def mean_response_time(response):
    """
    Plot the mean response time of all the participants and saves the plot
    Args:
        response: Response times of all participants
    Returns:
        None
    """

    # Calculate the mean response times
    mean_response_times = {p_id: np.mean(times) for p_id, times in response.items()}

    # Create lists for participant IDs and their mean response times.
    participant_ids = list(mean_response_times.keys())
    mean_times = list(mean_response_times.values())

    # Once all data is collected, plot the mean response times
    plt.figure(figsize=(60, 36))
    plt.bar(participant_ids, mean_times)
    plt.title('Bar Plot of Mean Participant Response Times')
    plt.xlabel('Participant ID')
    plt.ylabel('Mean Response Time')

    plt.savefig('mean_response_times.png')

    plt.show()


def top_response_times(participant_response):
    """
    Plot the mean response time of the top three faster and slower participants response time and saves the plot
    Args:
        participant_response: Response times of all participants
    Returns:
        None
    """

    # Calculate the mean response times
    mean_response_times = {p_id: np.mean(times) for p_id, times in participant_response.items() if times}

    # Sort the dictionary by response times
    # x[1] corresponds to value (the response times)
    sorted_response_times = sorted(mean_response_times.items(), key=lambda x: x[1])

    # get the three participants with shortest and longest response times
    shortest_response_times = sorted_response_times[:3]
    longest_response_times = sorted_response_times[-3:]

    print("Three participants with the shortest response times:")
    for p_id, time in shortest_response_times:
        print(f"Participant ID: {p_id}, Response Time: {time}")

    print("\nThree participants with the longest response times:")
    for p_id, time in reversed(longest_response_times):  # Reverse to get longest first
        print(f"Participant ID: {p_id}, Response Time: {time}")

    # Prepare data for the plot
    shortest_ids = [item[0] for item in shortest_response_times]
    shortest_times = [item[1] for item in shortest_response_times]
    longest_ids = [item[0] for item in reversed(longest_response_times)]  # Reverse to get longest first
    longest_times = [item[1] for item in reversed(longest_response_times)]  # Reverse to get longest first

    # Plot the shortest response times
    plt.figure(figsize=(10, 6))
    plt.bar(shortest_ids, shortest_times, color='blue')
    plt.title('Bar Plot of Shortest Mean Participant Response Times')
    plt.xlabel('Participant ID')
    plt.ylabel('Shortest Mean Response Time')

    plt.savefig('shortest_response_times.png')
    plt.show()

    # Plot the longest response times
    plt.figure(figsize=(10, 6))
    plt.bar(longest_ids, longest_times, color='red')
    plt.title('Bar Plot of Longest Mean Participant Response Times')
    plt.xlabel('Participant ID')
    plt.ylabel('Longest Mean Response Time')

    plt.savefig('longest_response_times.png')
    plt.show()


def response_to_average(response_times):
    """
    Plot the mean response time of all the participants in relation to the average of the group and saves the plot
    Args:
        response_times: Response times of all participants
    Returns:
        None
    """

    # Calculate the overall average response time
    overall_average_time = np.mean([time for times in response_times.values() for time in times])

    # Normalize the individual averages by subtracting the overall average
    normalized_response_times = {p_id: np.mean(times) - overall_average_time
                                 for p_id, times
                                 in response_times.items()
                                 if times}

    # Sort the participants by their normalized response times
    sorted_participant_ids = sorted(normalized_response_times, key=normalized_response_times.get)

    # Determine their corresponding response times
    sorted_response_times = [normalized_response_times[p_id] for p_id in sorted_participant_ids]

    # Create labels for the participants
    labels = [f'P{idx+1} ({round(time+overall_average_time, 2)})' for idx, time in enumerate(sorted_response_times)]

    # Plot the response times
    plt.figure(figsize=(60, 16))
    plt.bar(labels, sorted_response_times)
    plt.axhline(0, color='red', linewidth=2)
    plt.xticks(rotation=90)
    plt.title('Mean Participant Response Times Relative to Average', fontsize=40)
    plt.xlabel('Participants (Average Response Time)', fontsize=40)
    plt.ylabel('Normalized Response Time', fontsize=40)

    plt.savefig('response_to_average.png')
    plt.show()


def model_plot_history(hstory):
    """
    Plot and display the accuracy and error progressions over training epochs for the model training history.

    Args:
        hstory: The history of the trained model, typically returned by model.fit() in keras. Should contain history of
        binary accuracy and loss for both training and validation sets over epochs.

    Returns:
        None
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(hstory.history["binary_accuracy"], label="train accuracy")
    axs[0].plot(hstory.history["val_binary_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Binary Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Binary Accuracy Evaluation")

    # create error subplot
    axs[1].plot(hstory.history["loss"], label="train error")
    axs[1].plot(hstory.history["val_loss"], label="validation error")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Evaluation")

    plt.show()
