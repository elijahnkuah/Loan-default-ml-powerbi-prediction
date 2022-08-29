FROM python:3.8.8

# Create the user that will run the app
# RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /scorecard_api

# ARG PIP_EXTRA_INDEX_URL

# Install requirements, including from Gemfury
ADD ./scorecard_api
RUN pip install --upgrade pip
RUN pip install -r /scorecard_api/requirements.txt

# RUN chown -R ml-api-user:ml-api-user ./

# USER ml-api-user

# EXPOSE 8001

RUN python3 app.py
