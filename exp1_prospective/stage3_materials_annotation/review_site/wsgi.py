"""Production WSGI entry point for the Stage 3 annotation site."""

from exp1_prospective.stage3_materials_annotation.review_site.app import create_app


app = create_app()
