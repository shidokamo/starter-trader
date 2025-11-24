include .env
export

install:
	pipenv install

run:
	pipenv run python3 main.py

requirements:
	pipenv requirements --exclude-markers > requirements.txt

deploy:requirements
	gcloud functions deploy ${NAME} \
		--gen2 \
		--region=${REGION} \
		--runtime=python312 \
		--source=${PWD} \
		--entry-point=run_perp \
		--trigger-http \
		--set-env-vars BASE=${BASE} \
		--set-env-vars QUOTE=${QUOTE} \
		--set-env-vars FREQ=${FREQ} \
		--set-env-vars QUANTILE=${QUANTILE} \
		--set-env-vars DIP=${DIP} \
		--set-env-vars TAKE_PROFIT=${TAKE_PROFIT} \
		--set-env-vars POSITION_TIMEOUT_HOURS=${POSITION_TIMEOUT_HOURS} \
		--set-env-vars BUY_PARTIAL_FILLS_TIMEOUT_HOURS=${BUY_PARTIAL_FILLS_TIMEOUT_HOURS} \
		--set-env-vars WINDOW_HOURS=${WINDOW_HOURS} \
		--set-env-vars SPLIT_ORDERS=${SPLIT_ORDERS} \
		--set-env-vars BUY_ORDER_SPREAD=${BUY_ORDER_SPREAD} \
		--set-env-vars SELL_ORDER_SPREAD=${SELL_ORDER_SPREAD} \
		--set-env-vars LEVERAGE=${LEVERAGE} \
		--set-env-vars NO_BUY=${NO_BUY} \
		--set-env-vars NO_SELL=${NO_SELL} \
		--set-env-vars PROD=TRUE \
		--set-env-vars COMMENT=${COMMENT} \

log:
	gcloud functions logs read ${NAME} --gen2

job:
	gcloud scheduler jobs create http ${NAME} \
		--schedule="every 1 minutes" \
		--uri=https://${REGION}-${PROJECT}.cloudfunctions.net/${NAME} \
		--oidc-service-account-email=${SCHEDULER_SERVICE_ACCOUNT}
