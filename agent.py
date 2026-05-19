import pandas as pd
import joblib
from openai import OpenAI

client = OpenAI(
    api_key="SUA_API_KEY"
)

model = joblib.load(
    "model/modelo.pkl"
)

# ferramenta 1
def prever_modelo(dados):

    entrada = pd.DataFrame([dados])

    risco = model.predict(
        entrada
    )[0]

    prob = model.predict_proba(
        entrada
    )[0].max()

    return {
        "risco": int(risco),
        "confianca": f"{prob:.2%}"
    }


# ferramenta 2
def buscar_contexto(risco):

    if risco == 1:
        return """
        Casos semelhantes no dataset
        indicam maior probabilidade
        de doença cardíaca.
        """

    return """
    Casos semelhantes apresentam
    menor risco cardíaco.
    """


def agente(pergunta_usuario, dados, max_steps=5):

    mensagens = [

        {
            "role":"system",
            "content":
            """
            Você é um especialista em cardiologia.

            Use as ferramentas disponíveis.

            Explique em linguagem simples.
            """
        },

        {
            "role":"user",
            "content": pergunta_usuario
        }
    ]

    for _ in range(max_steps):

        resultado = prever_modelo(
            dados
        )

        contexto = buscar_contexto(
            resultado["risco"]
        )

        resposta = client.chat.completions.create(

            model="gpt-4o-mini",

            messages=[

                *mensagens,

                {
                    "role":"user",
                    "content":
                    f"""
                    Resultado do modelo:

                    {resultado}

                    Contexto:

                    {contexto}
                    """
                }
            ]
        )

        return resposta.choices[0].message.content

    return "Limite de passos atingido."