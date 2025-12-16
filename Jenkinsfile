pipeline {
    agent any

    environment {
        VENV_DIR     = 'venv'
        GCP_PROJECT  = 'mlops-new-447207'
        GCLOUD_PATH  = '/var/jenkins_home/google-cloud-sdk/bin'
    }

    stages {

        stage('Checkout Source Code') {
            steps {
                echo 'Checking out source code from GitHub...'
                // Default checkout (Jenkins handles SCM automatically)
            }
        }

        stage('Setup Virtual Environment & Install Dependencies') {
            steps {
                echo 'Setting up virtual environment and installing dependencies...'
                sh '''
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                '''
            }
        }
    }
}
