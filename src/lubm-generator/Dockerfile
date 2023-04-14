FROM maven:3.8.6-jdk-8
WORKDIR /app
COPY pom.xml /app/
COPY src/ /app/src/
RUN mvn package

# The .jar file can be built with:
# > docker build -t build-app .
# > docker run --rm -v "$(pwd)":/app build-app cp /app/target/lubm-uba.jar /app/
# It then needs to be placed into the ./target directory.