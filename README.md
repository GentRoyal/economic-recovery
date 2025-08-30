This project uses data from the FRED API, which requires an API key for access.

Steps to Get Started:
1. Sign up for a free FRED account
Go to: https://fred.stlouisfed.org
Create an account and log in.


2. Get your API key
Visit your account dashboard or go to https://fred.stlouisfed.org/docs/api/api_key.html
Copy the 32-character API key generated for you.

3. Create a .env file in the root of the project, enter the following
API_KEY=The 32-character API key generated for you
DB_NAME=data.sqlite