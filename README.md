# Assistente de voz local para Windows

Assistente de voz executado localmente, com Pipecat, Whisper via CUDA, LM Studio
e Piper TTS em português brasileiro. O atalho global `Shift+Espaço` e o menu da
bandeja alternam a sessão de voz.

## Requisitos

- Windows 11
- Python 3.13
- LM Studio com o servidor local iniciado em `http://localhost:1234/v1`
- GPU NVIDIA com CUDA para Whisper (ou configure o Whisper para CPU)
- Microfone e dispositivo de saída de áudio disponíveis no Windows

## Instalação

Na pasta do projeto, crie o ambiente virtual somente se ele ainda não existir
e instale as dependências:

```powershell
if (-not (Test-Path ".venv\Scripts\python.exe")) { uv venv --python 3.13 }
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Se `.venv` já existir, não é necessário executar `uv venv` novamente. Se o
comando perguntar se deve substituir o ambiente, responda `n` para mantê-lo.
Se respondeu `y`, o `uv` recriou apenas o ambiente virtual; instale novamente
as dependências com o segundo comando acima.

No LM Studio, carregue um modelo e inicie o servidor local. O ID do modelo é
lido de `config.toml`, na raiz do projeto:

```toml
[lm_studio]
model = "gemma4-12b-qat-uncensored-hauhaucs-balanced"
```

Edite o valor de `model` para corresponder exatamente ao ID apresentado pelo
servidor local do LM Studio. Não é necessário definir `LM_STUDIO_MODEL` no
terminal. O arquivo é carregado ao iniciar o aplicativo; reinicie-o depois de
alterar a configuração.

Outras configurações opcionais podem ser definidas por variáveis de ambiente:

| Variável | Padrão | Uso |
|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | URL local da API do LM Studio |
| `WHISPER_MODEL` | `small` | Modelo Faster-Whisper |
| `WHISPER_DEVICE` | `cuda` | Dispositivo Whisper; use `cpu` sem GPU compatível |
| `WHISPER_COMPUTE_TYPE` | `float16` | Tipo de cálculo; use `int8` em CPU |
| `PIPER_VOICE` | `pt_BR-faber-medium` | Voz Piper |

O código valida que a URL do LM Studio aponta para `localhost`, `127.0.0.1` ou
`::1`. Isso impede o envio das conversas para um endpoint remoto. Os modelos de
Whisper e Piper podem precisar ser baixados na primeira execução; depois de
armazenados em cache, a inferência ocorre localmente.

## Execução

Com console:

```powershell
.\.venv\Scripts\python.exe -m bin.main
```

Sem janela de console:

```powershell
.\.venv\Scripts\pythonw.exe -m bin.main
```

O ícone aparece na bandeja do sistema. Use `Shift+Espaço` ou **Iniciar/parar
sessão** para alternar a captura; escolha **Sair** no menu para encerrar.
Para execução automática sem janela, configure o Agendador de Tarefas do Windows
para iniciar `pythonw.exe` com o argumento `-m bin.main` e defina a pasta do
projeto como **Iniciar em**.

Os logs da aplicação ficam em `%LOCALAPPDATA%\WindAgent\logs`.

> **Distribuição:** a integração Piper em processo usa `piper-tts`, licenciado
> sob GPL-3.0. Avalie as obrigações da licença antes de redistribuir o
> aplicativo.
