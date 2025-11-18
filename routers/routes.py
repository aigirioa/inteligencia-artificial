from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from services.process_data import process_data, train_models

router = APIRouter()
templates = Jinja2Templates(directory='templates')


@router.get('/')
def index(request: Request):
    return templates.TemplateResponse('home.html', {'request': request})


@router.get('/process-models')
def process_models(request: Request):
    df, snapshots = process_data()

    random_forest_mse, linear_regression_mse = train_models(df)

    table_original = snapshots['original'].to_html(classes='table table-sm table-striped table-bordered', index = False)
    table_score = snapshots['score'].to_html(classes='table table-sm table-striped table-bordered', index = False)
    table_gender = snapshots['gender'].to_html(classes='table table-sm table-striped table-bordered', index = False)
    table_test_preparation_course = snapshots['test_preparation_course'].to_html(classes='table table-sm table-striped table-bordered', index = False)
    table_lunch = snapshots['lunch'].to_html(classes='table table-sm table-striped table-bordered', index = False)
    table_race_ethnicity = snapshots['race_ethnicity'].to_html(classes='table table-sm table-striped table-bordered', index = False)
    table_parental_level_of_education = snapshots['parental_level_of_education'].to_html(classes='table table-sm table-striped table-bordered', index = False)

    context = {
        'request': request,
        'table_original': table_original,
        'table_score': table_score,
        'table_gender': table_gender,
        'table_test_preparation_course': table_test_preparation_course,
        'table_lunch': table_lunch,
        'table_race_ethnicity': table_race_ethnicity,
        'table_parental_level_of_education': table_parental_level_of_education,
        'random_forest_mse': random_forest_mse,
        'linear_regression_mse': linear_regression_mse,
    }

    return templates.TemplateResponse('process_models.html', context)
