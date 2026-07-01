import pandas as pd
import matplotlib.pyplot as plt

students = pd.DataFrame({
    "Name": [
        "John", "Anna", "Mike", "Sara", "David", "Lina",
        "Tom", "Emma", "Chris", "Sophia", "James", "Olivia",
        "Daniel", "Grace", "Noah", "Mia", "Ethan", "Chloe",
        "Lucas", "Ella", "Samuel", "Hannah", "Nathan", "Ava",
        "Ryan", "Zoe", "Leo", "Ruby", "Adam", "Lucy"
    ],
    "Math": [
        80, 92, 68, 88, 75, 95,
        70, 85, 78, 91, 66, 83,
        74, 89, 97, 81, 72, 94,
        77, 86, 69, 90, 73, 84,
        79, 98, 76, 87, 71, 93
    ],
   "Physics":[
    45, 52, 58, 61,
    65, 68, 70, 71, 72,
    74, 75, 76, 77, 78, 78, 79,
    80, 81, 82, 82, 83, 84, 84, 85,
    86, 88, 90, 93, 96, 99
]

,
    "Gender": [
        "M", "F", "M", "F", "F", "F",
        "M", "F", "M", "F", "M", "F",
        "M", "F", "M", "F", "M", "F",
        "M", "F", "M", "F", "M", "F",
        "M", "F", "M", "F", "M", "F"
    ]
})

plt.hist(students["Physics"], bins=6, edgecolor = "Black")
plt.savefig("histogram.png")