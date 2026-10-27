# Assistente de voz local para Windows

Assistente de voz executado localmente, com Pipecat, Whisper via CUDA, provedores
de LLM compatíveis com a API OpenAI e os mecanismos Qwen3-TTS, Kokoro-82M ou
Piper TTS. O atalho
global `Shift+Espaço` e o menu da bandeja alternam a sessão de voz.

## Requisitos

- Windows 11
- Python 3.13
- Um provedor de LLM com API compatível com OpenAI (LM Studio, Ollama,
  OpenRouter, OpenAI ou outro compatível)
- GPU NVIDIA com CUDA para Whisper (ou configure o Whisper para CPU)
- Microfone e dispositivo de saída de áudio disponíveis no Windows

## Instalação

Na pasta do projeto, crie o ambiente virtual somente se ele ainda não existir
e instale as dependências:

```powershell
if (-not (Test-Path ".venv\Scripts\python.exe")) { uv venv --python 3.13 }
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe -r requirements-cuda.txt
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchaudio==2.7.0
uv pip install --python .venv\Scripts\python.exe -r requirements-qwen3.txt
```

`requirements-cuda.txt` instala as bibliotecas CUDA 12 e cuDNN 9 usadas pelo
Faster-Whisper na GPU NVIDIA. Elas ficam no ambiente virtual do projeto; o app
adiciona os diretórios dessas DLLs ao caminho de busca ao iniciar uma sessão.
Os comandos seguintes instalam PyTorch com CUDA 12.8 para a GPU NVIDIA RTX 50,
o Qwen3-TTS e suas dependências.

Se `.venv` já existir, não é necessário executar `uv venv` novamente. Se o
comando perguntar se deve substituir o ambiente, responda `n` para mantê-lo.
Se respondeu `y`, o `uv` recriou apenas o ambiente virtual; instale novamente
as dependências repetindo os comandos de instalação acima.

Os perfis de provedor ficam em `[llm.providers]` no `config.toml`. O menu da
bandeja mostra os perfis configurados; **Provedor de IA (próxima sessão)** salva
a escolha em `selected_provider`. Cada sessão usa somente o perfil selecionado.
Para LM Studio:

```toml
[llm]
selected_provider = "lm_studio"

[llm.providers.lm_studio]
base_url = "http://localhost:1234/v1"
model = "gemma4-12b-qat-uncensored-hauhaucs-balanced"
```

Para manter LM Studio, Ollama, OpenRouter e OpenAI disponíveis no menu ao mesmo tempo,
adicione os perfis necessários. As chaves são configuradas por perfil no próprio
arquivo, como solicitado:

```toml
[llm]
selected_provider = "openrouter"

[llm.providers.lm_studio]
base_url = "http://localhost:1234/v1"
model = "openai/gpt-oss-20b"

[llm.providers.ollama]
base_url = "http://localhost:11434/v1"
model = "qwen3:8b"

[llm.providers.openrouter]
base_url = "https://openrouter.ai/api/v1"
model = "openai/gpt-4o-mini"
api_key = "SUA_CHAVE_OPENROUTER"

[llm.providers.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key = "SUA_CHAVE_OPENAI"
```

O Ollama usa a API compatível com OpenAI em `http://localhost:11434/v1`. Inicie o
Ollama e baixe o modelo escolhido antes de iniciar a sessão; configure em `model`
o nome mostrado por `ollama list`. Ollama não precisa de `api_key`. O menu
apresenta somente os perfis definidos em `[llm.providers]`. Para usar outro
serviço local ou remoto compatível com OpenAI Chat Completions, adicione uma
tabela com o identificador desejado, `base_url`, `model` e `api_key` se exigir
autenticação. As chaves ficam em texto claro no `config.toml`; mantenha esse
arquivo privado. LM Studio pode omitir `model` para detectar o modelo carregado.
Reinicie o app após editar ou adicionar perfis; a seleção feita no menu é salva
imediatamente e vale na próxima sessão.

Outras configurações opcionais podem ser definidas por variáveis de ambiente:

| Variável | Padrão | Uso |
|---|---|---|
| `WHISPER_MODEL` | `small` | Modelo Faster-Whisper |
| `WHISPER_DEVICE` | `cuda` | Dispositivo Whisper; use `cpu` sem GPU compatível |
| `WHISPER_COMPUTE_TYPE` | `float16` | Tipo de cálculo; use `int8` em CPU |
| `PIPER_VOICE` | `pt_BR-faber-medium` | Voz Piper |
| `AUDIO_INPUT_DEVICE` | Entrada padrão do Windows | Substituição opcional por índice ou nome do dispositivo |
| `AUDIO_INPUT_GAIN` | `4.0` | Ganho fixo do microfone com proteção contra clipping; ajuste se necessário |

Por padrão, o Whisper usa CUDA na GPU NVIDIA. Se as DLLs CUDA necessárias não
estiverem disponíveis, o app informa isso no log e não inicia uma sessão de voz.
Para executar explicitamente em CPU, defina `WHISPER_DEVICE=cpu` e
`WHISPER_COMPUTE_TYPE=int8`.

LM Studio e Ollama são validados para apontar a `localhost`, `127.0.0.1` ou
`::1`. Ao selecionar um provedor remoto, transcrições e contexto da conversa são
enviados ao serviço indicado na configuração. Os modelos de Whisper e Piper podem
precisar ser baixados na primeira execução; depois de armazenados em cache, a
inferência ocorre localmente.

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
Para escolher a entrada, abra o menu do ícone do WindAgent na bandeja do Windows,
entre em **Microfone (próxima sessão)** e selecione o dispositivo. A opção marcada
é usada ao iniciar a próxima sessão e fica salva em
`%LOCALAPPDATA%\WindAgent\settings.json`. O menu mostra uma única entrada por
microfone que aceita a taxa de captura do Whisper, sem listar mapeadores do
Windows nem repetir cada entrada por driver. Sem escolha salva, usa a entrada
padrão do sistema.
Ao iniciar uma sessão, o log `application.log` informa quando o microfone
entrega áudio, quando o Whisper conclui uma transcrição e quando o provedor LLM
começa a responder. Se não aparecer a mensagem de áudio do microfone, selecione
o microfone correto como entrada padrão nas configurações de som do Windows.

No menu da bandeja, use **Idioma da conversa (próxima sessão)** para escolher
português do Brasil, inglês dos Estados Unidos ou espanhol. O idioma selecionado
configura a transcrição do Whisper e o idioma de resposta do assistente. O menu
**Voz [mecanismo] (próxima sessão)** mostra as vozes disponíveis para o idioma
selecionado e para o mecanismo ativo. Idioma e vozes selecionadas ficam salvos
na seção `[conversation]` de
`config.toml`; a seleção de microfone continua em
`%LOCALAPPDATA%\WindAgent\settings.json`. As escolhas de conversa passam a valer
na próxima sessão; os modelos de voz podem ser baixados na primeira utilização.

Em **Configuração da voz (próxima sessão)**, escolha velocidade lenta, normal
ou rápida; estilo estável, natural ou expressivo; e volume suave, médio ou alto.
O estilo ajusta a variação da síntese Piper, não controla diretamente o pitch.
As três preferências também ficam em `[conversation].voice` no `config.toml`.

Os prompts do sistema ficam em `[conversation.prompts]` no mesmo arquivo, com
uma chave para cada idioma (`pt`, `en` e `es`). Edite o texto da chave do idioma
que usa e reinicie o aplicativo para carregar o prompt personalizado.

Em **Voz TTS (próxima sessão)**, escolha Qwen3-TTS local, Kokoro-82M ou Piper.
O Kokoro-82M sintetiza em fluxo e oferece vozes próprias para português do
Brasil, inglês e espanhol; selecione-as em **Voz Kokoro (próxima sessão)**. Os
arquivos do modelo e das vozes são baixados automaticamente no primeiro uso
(cerca de 354 MB) e ficam em `%LOCALAPPDATA%\WindAgent\kokoro`. A velocidade
escolhida também se aplica ao Kokoro; estilo e volume configurados são usados
pelo Piper e pelo Qwen3.

Qwen3-TTS usa
a GPU NVIDIA quando `[tts.qwen3].device = "cuda"`; na primeira sessão baixa os
pesos do modelo Base 0.6B. Para usar clonagem de voz, indique um WAV em
`[tts.qwen3].reference_audio` e a transcrição correspondente em
`[tts.qwen3].reference_text`. Também é possível trocar o WAV no menu da bandeja
**Voz de referência Qwen3-TTS**. O prompt de voz processado é armazenado em
`%LOCALAPPDATA%\WindAgent\qwen3\voice_prompts` e reutilizado entre sessões.
Piper permanece disponível com vozes separadas por idioma.

> **Distribuição:** a integração Piper em processo usa `piper-tts`, licenciado
> sob GPL-3.0. Avalie as obrigações da licença antes de redistribuir o
> aplicativo.
