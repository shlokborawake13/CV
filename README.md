Got it 👍 You’ve uploaded a paper with Set 3 that has 3 DevOps-style tasks (GitHub, Jenkins, Docker).
Below are the answers/steps and expected outputs for each question — exactly what you’d need to perform and show in your practical or lab exam.


---

Q1. (15 Marks)

Task: Clone your GitHub repository, modify one Java file, commit and push the changes.
Expected Output: Updated code with new commit visible on GitHub.

✅ Steps / Answer

1. Clone the repository

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>


2. Modify a Java file Open any .java file in an editor and add a simple change, for example:

System.out.println("Updated version of the program!");


3. Check Git status

git status


4. Add and commit changes

git add .
git commit -m "Updated Java file with new message"


5. Push changes to GitHub

git push origin main


6. Verify output

Go to your GitHub repo in a browser.

You should see your new commit message and updated code.




✅ Expected Output:

> GitHub repository shows the latest commit — “Updated Java file with new message”.




---

Q2. (15 Marks)

Task: Create a Jenkins pipeline with stages: Checkout → Build → Test.
Expected Output: Pipeline stages execute successfully and console output shown.

✅ Steps / Answer

1. Create a new Jenkins pipeline project

Open Jenkins → “New Item” → choose Pipeline → name it JavaApp-Pipeline.



2. In the pipeline configuration, add the following Pipeline script:

pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/<your-username>/<your-repo>.git'
            }
        }

        stage('Build') {
            steps {
                sh 'javac src/*.java'
            }
        }

        stage('Test') {
            steps {
                sh 'echo Running tests...'
                sh 'java org.junit.runner.JUnitCore MyTestClass'  // Example if you have tests
            }
        }
    }
}


3. Save and build the pipeline.


4. Observe Jenkins console output:

Stages: Checkout, Build, Test appear.

No errors are shown.




✅ Expected Output:

> Jenkins console shows successful execution of all three stages:
✔ Checkout completed
✔ Build completed
✔ Test completed




---

Q3. (10 Marks)

Task: Build a Docker image for your Java project and verify its creation.
Expected Output: Image listed in Docker images list.

✅ Steps / Answer

1. Create a Dockerfile in your project directory:

FROM openjdk:17
WORKDIR /app
COPY . /app
RUN javac src/Main.java
CMD ["java", "src/Main"]


2. Build the Docker image

docker build -t javaapp:latest .


3. Verify image creation

docker images


4. Run the container (optional)

docker run javaapp:latest



✅ Expected Output:

REPOSITORY      TAG       IMAGE ID       CREATED          SIZE
javaapp         latest    <some-id>      a few seconds ago  <size>


---

✅ Final Summary (for report or practical submission)

Q.No	Task Summary	Expected Output

Q1	GitHub clone, modify Java file, commit, push	Updated commit visible on GitHub
Q2	Jenkins pipeline: Checkout → Build → Test	Console shows successful stages
Q3	Docker build and image verification	Image listed in docker images



---

Would you like me to create a PDF file of these answers formatted like an exam submission (with proper headings and outputs)?

# CV