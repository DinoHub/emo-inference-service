build:
	docker build . -t gradio-emo-inference-service:1.0.0
dev:
	docker run -p 8083:8083 --rm -it -v ${PWD}:/demo gradio-emo-inference-service:1.0.0
