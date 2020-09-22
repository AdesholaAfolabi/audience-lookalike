# audience-lookalike
This README documents the steps that are necessary to get the lookalike application up and running. The lookalike application is an unsupervised machine learning  algorithm that helps to identify audiences that are similar to a seed audience (usually the replica you want to create) for digital campaign purposes.

### What is this repository for? ###

* Quick summary: The application employs the modular style of putting the applications together. In total, there are four modules which takes care of pre-processing, modeling, and scoring. There is a general module that contains a list of global attributes and data used throughout the project.
* Version: 1.0
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up: ensure you have python 3 up and running
* Configuration: ensure all modules are imported properly. They all depend on each other
* Dependencies: python 3, pandas, scikit-learn, sklearn pre-processing
* Database configuration: no required configuration
* How to run tests: no tests files used yet. Version 2 will come with test cases
* Deployment instructions: call the score module and include a file path. That will return scores of MSISDNs and their cluster distances.

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner: afolabimkay@gmail.com
