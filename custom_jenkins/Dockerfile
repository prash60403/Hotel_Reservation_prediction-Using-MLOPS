# Jenkins base image
FROM jenkins/jenkins:lts

# Switch to root
USER root

# Install Docker CLI only (NOT docker-ce daemon)
RUN apt-get update && \
    apt-get install -y docker.io && \
    apt-get clean

# Allow Jenkins user to run Docker
RUN usermod -aG docker jenkins

# Switch back to Jenkins user
USER jenkins
