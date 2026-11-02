# Como rodar o treino (PC Windows, RTX 2060)

Tudo em **PowerShell**. Rode na ordem. Não precisa instalar Python nem CUDA Toolkit: o `uv sync` baixa o Python 3.12 e o torch com CUDA 12.6. Só o driver NVIDIA é necessário.

## 1. Pré-requisitos (uma vez)

```powershell
nvidia-smi                      # deve mostrar a RTX 2060 e a versão do driver
git --version
git config user.name            # se vazio: git config --global user.name "Seu Nome"
git config user.email           # se vazio: git config --global user.email "voce@exemplo.com"
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Feche e abra o PowerShell depois de instalar o uv e confira:

```powershell
uv --version
```

Para o `git push` funcionar, é preciso estar logado no GitHub (Git Credential Manager abre o navegador no primeiro push; ou `gh auth login`).

## 2. Clonar (caminho sem espaços)

```powershell
cd C:\
git clone https://github.com/<usuario>/cyclegan-mapas-satelite.git
cd C:\cyclegan-mapas-satelite
```

Troque `<usuario>` pelo dono do repositório. Se já clonou antes: `git pull`.

## 3. Copiar os dados

O `dados.zip` (553 MB) **não está no git**. Copie por pendrive/Drive para `C:\cyclegan-mapas-satelite\dados.zip` e extraia:

```powershell
Expand-Archive dados.zip -DestinationPath data
Get-ChildItem data
```

Deve listar exatamente `trainA`, `trainB`, `testA`, `testB`. Se aparecer uma pasta extra no meio (ex.: `data\dados\trainA`), mova o conteúdo para cima:

```powershell
Move-Item data\dados\* data\ ; Remove-Item data\dados
```

Conferência de contagem (esperado: 6099, 6586, 1056, 947):

```powershell
foreach ($d in "trainA","trainB","testA","testB") { "$d : " + (Get-ChildItem data\$d).Count }
```

## 4. Ambiente e teste da GPU

```powershell
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Esperado: `True NVIDIA GeForce RTX 2060`. Se for `False`, veja "Problemas comuns".

## 5. Teste de velocidade e de resume (3-4 min)

Roda 200 iterações, depois repete com `--resume` para validar também a retomada na GPU:

```powershell
uv run python train.py --data data --run velocidade --epochs 1 --max-iters 200
uv run python train.py --data data --run velocidade --epochs 2 --max-iters 200 --resume
```

O segundo comando deve imprimir `Retomando da época 1 (próxima: 2).` e criar `runs\velocidade\amostras\epoca_002.jpg`. Abra as grades `epoca_001.jpg` e `epoca_002.jpg` e confira que estão coerentes (imagens reais | traduzidas | reconstruídas, sem lixo/tela preta).

Leia no terminal, na **última** linha de log (`it 200/200`):

- **it/s** (iterações por segundo): quanto maior, melhor. A 1ª linha (`it 50`) inclui a criação dos workers e o `cudnn.benchmark`, então é pessimista: ignore-a. Também **não** use o `its_por_s` do `epocas.csv` para a conta abaixo (inclui esse início lento).
- **VRAM** (GB): é o `max_memory_allocated` do PyTorch, não inclui o cache do alocador nem o uso do desktop. Compare com o `nvidia-smi` em **outra** janela durante o teste; ~5 GB no `nvidia-smi` é o teto confortável para a RTX 2060 (6 GB).

Tempo por época = 6586 / it/s (do `it 200/200`). Conta de cabeça: a 7 it/s, 6586 / 7 ≈ 940 s ≈ 16 min.

Estimativa grosseira para a RTX 2060 com AMP: **~15–20 min/época, ou seja, ~25–35 épocas numa noite de ~9 h**. É uma estimativa: confirme com o teste e refaça a conta com o seu it/s.

Só depois de conferir a grade, apague o teste:

```powershell
Remove-Item -Recurse -Force runs\velocidade
```

**Para repetir o teste:** apague a pasta `runs\velocidade` ou use outro `--run`. O `train.py` aborta se `last.pt` já existir sem `--resume`.

## 6. Preparar o PC para a noite

Não deixar suspender (com o carregador ligado):

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

Também:

- Windows Update: Configurações > Windows Update > **Pausar atualizações**.
- Mantenha o PC na tomada e o terminal **aberto** (não feche a janela, não faça logoff).
- Feche navegador e jogos para sobrar VRAM.

Para reverter depois (ex.: 30 minutos):

```powershell
powercfg /change standby-timeout-ac 30
powercfg /change hibernate-timeout-ac 30
```

## 7. Treino principal

```powershell
uv run python train.py --data data --run principal --epochs 100 --decay-start 50 --snapshot-every 5
```

### Por que 100 épocas se só dá para fazer ~30 numa noite

O agendamento de LR é **fixo**: constante até a época 50, depois cai linearmente até quase zero na 100. Ele depende só do número da época, não do que já rodou. Então não precisa terminar tudo de uma vez: o treino pode ser **interrompido (Ctrl+C) e retomado a qualquer hora** com `--resume`, e o LR continua exatamente de onde deveria. Na primeira noite, o LR ainda está constante (épocas < 50), o que é esperado.

### O que aparece no terminal

- A cada 50 iterações: `ép E/T it I/N | X it/s | ETA ép M min | VRAM G GB | G_adv a/b cyc c/d D e/f` (perdas = médias da janela: `G_adv_AB/BA`, `cyc_A/B`, `D_A/D_B`). it/s, ETA e VRAM também ficam em `epocas.csv`.
- Ao fim de cada época: `== época N concluída em X min (lr ...); last.pt salvo.`
- Valores típicos de `D_A` e `D_B` no LSGAN ficam abaixo de ~0,5. `D` colado em 0 por muitas épocas, com `G_adv` subindo sem parar, é sinal de alerta (o D está ganhando demais).
- `cyc_A` e `cyc_B` devem **cair** nas primeiras épocas.

**Não abra os CSV no Excel durante o treino** (o Windows trava o arquivo e o `PermissionError` derruba o treino). Mantenha a pasta do repositório **fora do OneDrive** ou de qualquer pasta sincronizada pelo mesmo motivo. `Get-Content -Tail` é seguro.

### Onde ver o progresso

- `runs\principal\amostras\epoca_XXX.jpg`: grade com imagens de teste fixas (real | traduzida | reconstruída), uma por época. Abra a mais recente para ver a evolução.
- `runs\principal\epocas.csv` e `perdas.csv`: números.
- `runs\principal\G_epoca_XXX.pt`: snapshots a cada 5 épocas (ficam só no PC). Servem para inferência (só os geradores), **não** para `--resume`.

Uma mini-checagem no meio da noite, em **outro** terminal:

```powershell
Get-Content runs\principal\epocas.csv -Tail 3
nvidia-smi
```

## 8. De manhã: parar e enviar

No terminal do treino:

1. **Ctrl+C**. Ele avisa qual foi a última época salva. Uma época incompleta é descartada (o `last.pt` só é gravado ao fim de cada época).
2. Enviar para o GitHub (os `.pt` são ignorados pelo `.gitignore`, só vão CSVs, `config.json` e imagens):

```powershell
git add runs/principal
git commit -m "feat: resultados do treino principal até a época N"
git push
```

Troque `N` pela última época. Confira com `git status` que não há `.pt` na lista antes do commit.

3. Se quiser, restaure a suspensão (seção 6).

## 9. Retomar o treino (próximas noites)

Mesmo comando, mais `--resume`. **Não mude** `--epochs`, `--decay-start`, `--lr` nem `--run`.

```powershell
uv run python train.py --data data --run principal --epochs 100 --decay-start 50 --snapshot-every 5 --resume
```

Ele lê `runs\principal\last.pt` e continua na época seguinte. Linhas do `perdas.csv` de uma época incompleta anterior são removidas antes de continuar. As imagens fixas das grades ficam as mesmas (vêm do `config.json`).

Se o código foi alterado no repositório desde a última noite, rode `git pull` antes de retomar.

## 10. Ablação (identity loss)

Run curta, para comparar com o principal. Faça depois do principal (ou numa noite separada):

```powershell
uv run python train.py --data data --run ablacao_id --lambda-id 0.5 --epochs 20 --decay-start 50
```

Como `--decay-start 50` é maior que `--epochs 20`, o LR fica **constante** nas 20 épocas, que é comparável com as primeiras épocas do principal. Para enviar: `git add runs/ablacao_id`, commit e push, como na seção 8. Para retomar, adicione `--resume`.

## Acompanhar o treino pelo celular (opcional)

O `monitor.py` roda num **segundo terminal**, com o treino já iniciado, e manda uma mensagem para o celular a cada 30 min: épocas concluídas, época e iteração atuais, perdas, min/época, horas restantes e temperatura/VRAM da GPU. Ele também alerta **na hora** se o treino travar (10 min sem progresso), se as perdas virarem NaN ou quando o treino terminar.

1. No celular, instale o app **ntfy** (Android/iOS) e assine um tópico com um nome longo e aleatório, por exemplo `ganmapas-x7k2q9w4m1` (quem souber o nome consegue ler as mensagens, então não use algo adivinhável).
2. No PC, em outro terminal na pasta do projeto:

```powershell
uv run python monitor.py --run principal --topic ganmapas-x7k2q9w4m1
```

Na primeira execução chega uma mensagem na hora, o que confirma que o celular está recebendo. Sem `--topic`, ele só imprime no terminal. Use `--every 15` para mudar o intervalo e `--once` para testar.

Se as mensagens pararem de chegar sem nenhum alerta, o PC suspendeu, caiu a internet ou o próprio monitor foi fechado: vale ir ver.

## Problemas comuns

**`torch.cuda.is_available()` dá `False` / treino lento em CPU**
- Rode `nvidia-smi`. Se falhar, atualize o driver NVIDIA e reinicie.
- Rode `uv sync` de novo dentro da pasta do projeto (precisa baixar o torch `+cu126`). Confira com `uv run python -c "import torch; print(torch.__version__)"`: deve terminar em `+cu126`.
- Notebook com GPU híbrida: configure o Windows para usar a NVIDIA em "Configurações gráficas".
- Se o it/s do teste de velocidade for < 1, está rodando em CPU.

**CUDA out of memory**
- Feche outros programas que usam a GPU (navegador, jogos, OBS). Veja o uso em `nvidia-smi`.
- Reinicie o terminal e tente de novo. O treino usa batch 1 e AMP; se persistir, anote a VRAM máxima do teste de velocidade e avise o grupo.

**Erro de multiprocessing / DataLoader / `BrokenPipeError`**
- Rode com `--workers 0` (mais lento, mas não usa processos extras). Se for repetir o teste de velocidade, apague `runs\velocidade` antes (ou use outro `--run`): o `train.py` aborta se `last.pt` existir sem `--resume`.

**Perdas viram `nan` / `inf`**
- Se as perdas ficarem não finitas por 3 janelas de log seguidas, o treino aborta sozinho com `Perdas não finitas por 3 janelas seguidas: abortando. Retome com --resume --no-amp.`
- Retome com `--resume --no-amp` (fp16 desligado; mais lento e usa mais VRAM). O `last.pt` só é gravado ao fim de épocas completas, então a retomada parte da última época boa.
- Se a run já gravou `nan` em épocas salvas, apague-a (`Remove-Item -Recurse -Force runs\principal`) e comece de novo, ou retome de uma cópia anterior do `last.pt`, se tiver.

**O PC reiniciou / travou / faltou energia no meio**
- Nada está perdido: rode o comando da seção 9 (com `--resume`). Você perde, no máximo, a época em andamento (~15–20 min).
- O `last.pt` só é substituído depois de gravado por completo (o `last.pt.tmp` é parcial: não use). Se o `last.pt` mesmo assim não abrir, avise antes de apagar qualquer coisa; os snapshots `G_epoca_XXX.pt` servem só para inferência, não para retomar.
- Depois disso, refaça a seção 6 (o Windows pode ter voltado às configurações antigas).

**`git push` pede senha / é rejeitado**
- Faça login no GitHub (`gh auth login` ou o navegador do Git Credential Manager).
- Se rejeitado por estar atrás do remoto: `git pull --rebase` e `git push`.

**`uv` não é reconhecido**
- Feche e reabra o PowerShell depois de instalar. Se persistir, adicione `%USERPROFILE%\.local\bin` ao PATH.
