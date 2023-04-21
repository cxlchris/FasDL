FROM public.ecr.aws/lambda/python:3.8

RUN pip install opencv-python-headless

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]
