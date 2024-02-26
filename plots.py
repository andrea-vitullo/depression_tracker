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
    plt.figure(figsize=(60, 36))  # Creates a new figure
    plt.bar(participant_ids, mean_times)  # Plots mean participant response times
    plt.title('Bar Plot of Mean Participant Response Times')
    plt.xlabel('Participant ID')
    plt.ylabel('Mean Response Time')

    # Save the figure before showing it
    plt.savefig('mean_response_times.png')  # Provide the filename (.png, .jpg, .pdf, etc.)

    plt.show()  # Display the plot


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

    # Save the figure before showing it
    plt.savefig('shortest_response_times.png')  # Provide the filename (.png, .jpg, .pdf, etc.)
    plt.show()

    # Plot the longest response times
    plt.figure(figsize=(10, 6))
    plt.bar(longest_ids, longest_times, color='red')
    plt.title('Bar Plot of Longest Mean Participant Response Times')
    plt.xlabel('Participant ID')
    plt.ylabel('Longest Mean Response Time')

    # Save the figure before showing it
    plt.savefig('longest_response_times.png')  # Provide the filename (.png, .jpg, .pdf, etc.)
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
    plt.axhline(0, color='red', linewidth=2)  # add a line at zero for average
    plt.xticks(rotation=90)
    plt.title('Mean Participant Response Times Relative to Average', fontsize=40)
    plt.xlabel('Participants (Average Response Time)', fontsize=40)
    plt.ylabel('Normalized Response Time', fontsize=40)

    # Save the figure before showing it
    plt.savefig('response_to_average.png')  # Provide the filename (.png, .jpg, .pdf, etc.)
    plt.show()
