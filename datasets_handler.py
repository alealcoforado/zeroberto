import datetime
dict_classes_folha = {
    'poder':"Poder e Política no Brasil",
    'mercado':"Mercado",
    'mundo':"Notícias de fora do Brasil",
    'esporte':"Esporte",
    'tec':"Tecnologia",
    'ambiente':"Meio Ambiente",
    'equilibrioesaude':"Equilíbrio e Saúde",
    'educacao':"Educação",
    'tv':"TV, Televisão e Entretenimento",
    'ciencia':"Ciência",
    'turismo':"Turismo",
    'comida':"Comida" }

def getAgora():
    return str(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))