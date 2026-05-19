from agent import agente

paciente = {

    "age":52,
    "sex":1,
    "cp":0,
    "trestbps":125,
    "chol":212,
    "fbs":0,
    "restecg":1,
    "thalach":168,
    "exang":0,
    "oldpeak":1.0,
    "slope":2,
    "ca":2,
    "thal":3
}

resposta = agente(

    "Este paciente apresenta risco cardíaco?",

    paciente
)

print("\nResposta:\n")
print(resposta)