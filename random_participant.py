import random

random.seed(4122)

participant_numbers = list(range(1, 15))
conditions = ["pseudo first", "grid first"]
# attribute condition to each participant randomly

for i in range(1, 15):
    condition = random.choice(conditions)
    participant_numbers[i - 1] = (participant_numbers[i - 1], condition)

# save in txt file
with open("participant_numbers.txt", "w") as f:
    for i in range(1, 15):
        f.write(str(participant_numbers[i - 1]) + "\n")
