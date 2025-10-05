# Web based behavioral closed loop

This is a tutorial on running web based behavioral experiments with firebase and autora.

There are to environments to set up for the closed loop:

# Researcher Hub - Autora
This is the python scripts that will be run on a server to run the closed loop (this typically consists of an autora-experimentalist, autora-runner and autora-theorist)
Follow the steps here: [Set up Autora Workflow](researcher_hub/README.md)

# Testing Zone - Firebase
This is the website that is served to the participant. We use Firebase to host the website and Firestore as a database. The database gets populated with conditions from the autora-runner and stores observations when participants attend the website. The autora-runner will read the observations and pass them to the theorist.
Follow the steps here: [Set up Firebase](testing_zone/README.md)