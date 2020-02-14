# fake_data_analysis_with_ensembled_hybrid_algorithm

I am hosting this project in git to help people who are interested in using the research, feel free to use or reproduce it.

# Key Take Away of the Work
We know deep learning is powerful, but it comes with a cost of long training time. 
Simple machine learning algorithms are also great but can't do  the complete job with at most accuracy, but they are fast.

#OBSERVATION
Different simple ML algorithms are doing great at identifying different things correctly, on using 7 different algorithms which are
around 80% accuracy, i wanted to see how these things work as a cluster 

# The Project
so i clubed all the five and created a new algorithm which works based on the voting mechanism where all the 7 algo's vote for the
classification, the side which got more number of votes win.

# Result Impact
That's a significant impact, i raised from 80 to ~94 that's a dream accuracy with the usage of a just ML for a project like fake data
detection, the data is news here, but they were preclassified articles so I referred them as data. 

Please find the result metrix at the end.

# Structure
leaving the folder named "algorithm training" everything is a Django project, I used Django to create a communication mechanism, where 
you can provide some data and see whether that's real or fake.

you can find the code for training the classifier in "algorithm training" folder.

I pickled the result and used that as the data source in the Django application, I have already kept that pickle file in position
you can remove the "algorithm training" folder and start the Django server to try out the application.

# Readme is not enough
Really the entire research cannot be written in one readme file, this is huge so I want to provide the entire "documentation" I made to
clearly make you understand the findings.
# Note
The documentation is centered on the Django application so find out what you really want and go for that, don't spend time on areas what
you don't need.
