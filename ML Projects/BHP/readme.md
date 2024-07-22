<img width="940" alt="BHP" src="https://github.com/user-attachments/assets/29e89a12-a4f0-44bb-99a3-d3c0d72ab11c">
This data science project series walks through the step-by-step process of building a real estate price prediction website.

First, we will build a model using sklearn and linear regression with the Bangalore home prices dataset from kaggle.com. The second step involves writing a Python Flask server that uses the saved model to serve HTTP requests. The third component is a website built with HTML, CSS, and JavaScript that allows users to enter home details such as square footage and the number of bedrooms, and then calls the Python Flask server to retrieve the predicted price.

During model building, we will cover various data science concepts, including data loading and cleaning, outlier detection and removal, feature engineering, dimensionality reduction, gridsearchcv for hyperparameter tuning, and k-fold cross-validation.

In terms of technology and tools, this project covers:

- Python
- Numpy and Pandas for data cleaning
- Matplotlib for data visualization
- Sklearn for model building
- Jupyter Notebook, Visual Studio Code, and PyCharm as IDEs
- Python Flask for the HTTP server
- HTML/CSS/JavaScript for the UI

# Deploy this app to cloud (AWS EC2)
In order to do this I followed these ateps. I created EC2 instance using amazon console, also in security group added a rule to allow HTTP incoming traffic.
Finally I connected to the instance.


1. Create EC2 instance using amazon console, also in security group added a rule.
2. connecte to the instance:
    ```bash
    ssh -i "C:\Users\matin\.ssh\Banglore.pem" ec2-13-60-162-161.eu-north-1.compute.amazonaws.com
    ```

3. nginx setup
    1. Installed nginx on EC2 instance:
        ```bash
        sudo apt-get update
        sudo apt-get install nginx
        ```
    2.  Restart and start nginx:
        ```bash
        sudo service nginx start
        sudo service nginx stop
        sudo service nginx restart
        ```

4. copy all of my code to the EC2 instance. I used WinSCP.
5. copy all code files into the `/home/ubuntu/` folder. 
6. After copying code on the EC2 server, now it is possible to point nginx to load our property website by default.
    1. Create this file `/etc/nginx/sites-available/bhp.conf`. The file content looks like this:
        ```nginx
        server {
            listen 80;
            server_name bhp;
            root /home/ubuntu/BangloreHomePrices/client;
            index app.html;
            location /api/ {
                rewrite ^/api/(.*) $1 break;
                proxy_pass http://127.0.0.1:5000;
            }
        }
        ```
    2. Create a symlink for this file in `/etc/nginx/sites-enabled` by running this command:
        ```bash
        sudo ln -v -s /etc/nginx/sites-available/bhp.conf
        ```
    3. Remove the symlink for the default file in `/etc/nginx/sites-enabled` directory:
        ```bash
        sudo unlink default
        ```
    4. Restart nginx:
        ```bash
        sudo service nginx restart
        ```
At the end, it is sufficient to refresh or copy the URL in the search engine, and you will see that it works well and can predict the price efficiently.
