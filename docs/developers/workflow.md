# Development Workflow Guide

This document will guide you through our development workflow. It includes instructions on how to use our version control system, how to create and manage branches, how to handle issues, and how to submit your code for review.

## Version Control

We use Git for version control. All our code repositories are hosted on GitLab.

- To clone a repository, use the `git clone` command followed by the URL of the repository.
- To pull the latest changes from the main branch, use the `git pull origin main` command.

## Branching Strategy

See [Branch, Merge, Release Strategy](https://gitlab.jatic.net/jatic/docs/sdp/-/blob/main/Branch%2C%20Merge%2C%20Release%20Strategy.md).

To create a new branch, use the `git checkout -b [branch-name]` command.

### GitFlow Branching Strategy

**Master Branch:** This branch represents the official release history. It's meant to be stable, with new commits only coming from release branches. Every commit on this branch should be a new release, involving incrementing the version numbers and creating a tag.

**Develop Branch:** This is the development branch where the latest developments integrate. This is where developers collaborate and is the core of the GitFlow model.

**Feature Branches:** These branches are used to develop new features for the upcoming or a distant future release. When the feature is complete, it's merged into the develop branch. Feature branches usually originate from the latest state of the develop branch.

**Release Branches:** These branches are used to prepare for a new production release. They allow for last-minute dotting of i’s and crossing t's. When the release is ready to ship, it gets merged to master and tagged with a version number. In addition, it should be merged back into develop, which may have progressed since the release was initiated.

**Hotfix Branches:** These branches are used to quickly patch production releases. This is the only branch that should fork directly off of master. As soon as the fix is complete, it should be merged into both master and develop (or the current release branch), and master should be tagged with an updated version number.


## Issue Handling

We track all our tasks, bugs, and feature requests using GitLab Issues. Each issue is assigned to a specific person.

When you start working on an issue:

1. Assign the issue to yourself.
2. Create a new branch.
3. Once you've completed your work, create a merge request and link it to the issue.

## Code Reviews

Once you've completed your work on a branch, you'll need to submit a merge request (MR) for code review.

1. Push your branch to GitLab using the `git push origin [branch-name]` command.
2. Go to our repository on GitLab, where you'll see your recently pushed branches.
3. Click the 'Compare & merge request' button next to your branch.
4. Fill out the MR form. Include a summary of your changes and any issues the PR closes.
5. Assign at least one team member as a reviewer.
6. Once your MR is approved, merge your changes into the main branch.

Remember to always pull the latest changes from the main branch before starting to work on a new feature or fix.

Please don't hesitate to ask if you have any questions about our workflow. We're here to help!
