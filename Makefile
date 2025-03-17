PDF_PATH=./data/tdr_0.pdf

export_env:
	set -a 
	source .env
	set +a
update_packages:
	pip install -r requirements.txt
check_pdf:
	xdg-open $(PDF_PATH)