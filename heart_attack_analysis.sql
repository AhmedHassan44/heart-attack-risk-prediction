CREATE DATABASE IF NOT EXISTS heart_attack_analysis;

USE heart_attack_analysis;

CREATE TABLE IF NOT EXISTS heart_attack_data (
    age INT,
    sex VARCHAR(10),
    cp INT,
    trtbps INT,
    chol INT,
    fbs INT,
    restecg INT,
    thalachh INT,
    exng INT,
    oldpeak FLOAT,
    slp INT,
    caa INT,
    thall INT,
    output INT
);

