# Prompt para continuar o trabalho no notebook

> Cole este arquivo inteiro no Claude Code, aberto na pasta do repositório clonado.

Você vai me ajudar a terminar um trabalho de faculdade que já está quase pronto. **Não retreine o modelo.** O treino terminou em outro PC. Aqui só faltam a avaliação no conjunto de teste e o acabamento dos slides e do README.

## Contexto

- **Disciplina:** Aprendizado Não-Supervisionado (PUCRS, Prof. Luan Garcia). Grupo: João Henrique da Luz, Vittor Brits e Marcelo.
- **Entrega:** quinta-feira, 19/11/2026. O que se entrega é o **material de apresentação** (slides, até 20 min).
- **Projeto:** CycleGAN em PyTorch, feita do zero.
  - Domínio **A**: recortes de mapas históricos (David Rumsey).
  - Domínio **B**: ortoimagens de satélite (USGS).
  - Imagens de 256×256, sem pares.
- **Repositório:** https://github.com/Joao-H-Luz/cyclegan-mapas-satelite (público), branch **`main`**, que já tem tudo. O `README.md` explica a implementação, e o `COMO_RODAR.md` explica o treino.

### O que o enunciado pede (e os pesos)

| Critério | Peso | O que é avaliado |
|---|---|---|
| Domínios e dataset | 10% | Par justificado, tamanho, **como foi coletado**, pré-processamento, exemplos |
| Implementação | 10% | Gerador, discriminador e custos justificados |
| Treinamento | 20% | Épocas, LR, custos, tempo por época, tempo total, curvas adversarial e cíclica |
| Resultados | 20% | Imagens de teste, mesma quantidade de A e B, casos bons e ruins, evolução ao longo do treino |
| Análise crítica | 25% | Falhas explicadas com hipótese e evidência, limitações |
| Apresentação | 15% | Clareza, tempo, domínio do conteúdo |

A nota **não** depende da qualidade visual das imagens, e sim da análise.

## O que já está pronto

- **Treino:** completo, 100 épocas.
  - Tempo: 40,1 h somadas, 24,1 min por época (~4,6 it/s), VRAM de pico 5,0 GB, RTX 2060.
  - Hiperparâmetros: Adam lr 2e-4 com β (0,5; 0,999), batch 1, AMP fp16, λ_cyc = 10, λ_id = 0.
  - LR: constante até a época 50, depois decai linearmente até ~0 na 100.
- **No git:**
  - Todo o código: `models.py`, `train.py`, `analyze.py`, `avaliar.py`, `monitor.py` e `tests/`.
  - Os arquivos da run em `runs/principal/`: `config.json`, `epocas.csv`, `perdas.csv`, `amostras/epoca_001.jpg` a `epoca_100.jpg` e as figuras `evolucao_A2B.jpg` e `evolucao_B2A.jpg`. As figuras são 4 + 4 imagens fixas de teste, nas épocas 1, 5, 10, 25, 50, 75 e 100, mais a reconstrução final.
  - O README com a seção **Resultados e análise** preenchida.
- **Slides em HTML:** `apresentacao.html`, na raiz do repo, abre offline no navegador. É uma cópia exportada do Artifact abaixo. Depois de editar o Artifact, gere o HTML de novo: junte `project/deck.css` e os `project/slides/*.html` na ordem do `project/deck.json`, troque cada `/_blob/<id>` pela imagem local em base64 e mantenha o script de navegação do HTML atual.
- **Slides (17):** https://claude.ai/artifact/RmZJ1rSnTaerpTkLfUQH9p. É um Artifact do tipo Slides, na mesma conta do claude.ai. Para editar, use a ferramenta Artifact: primeiro `read` do deck e dos slides que vai mudar, depois publique só os arquivos alterados, em `project/slides/<id>.html`. Os ids dos slides: `cover, traducao, dados, coleta, arquitetura, custos, treino, curvas-adv, curvas-cic, evolucao-a2b, evolucao-b2a, analise-texto, analise-disc, analise-estego, analise-instab, limitacoes, conclusao`.
- **Análise crítica já escrita**, com 4 falhas, cada uma com evidência e hipótese:
  1. O texto do mapa é copiado (A→B) ou inventado (B→A).
  2. Mapa → satélite é o sentido mais difícil: D_B cai de 0,20 para 0,01, enquanto G_adv A→B sobe de 0,45 para 0,96.
  3. As reconstruções são quase perfeitas mesmo com tradução ruim. A hipótese é esteganografia (Chu et al., 2017).
  4. Instabilidades pontuais: faixas coloridas na época 5 (B→A), picos de G_adv B→A nas épocas 61–65 e 83–84, e manchas rosa na época 100.

## O que falta fazer, em ordem

### 1. Preparar o ambiente

- `uv sync`. Se não tiver o `uv`, instale: https://docs.astral.sh/uv/. O `pyproject.toml` define o torch.
- O snapshot final `runs/principal/G_epoca_100.pt` (44 MB, só os geradores) **já vem no git**. É o único `.pt` versionado.
- Arquivos que **não estão no git**:
  - `data/testA` e `data/testB`, que vêm do `dados.zip` (por pendrive ou Drive). Confira as contagens: **1056** e **947** imagens.
  - Os dados de treino **não** são necessários.
- Se o Python for bloqueado pelo Windows (erro 4551, "política de Controle de Aplicativo" = Smart App Control), **pare e avise o usuário**. Não tente contornar.

### 2. Rodar os testes primeiro

```
uv run python -m pytest -q
```

**O `avaliar.py` nunca foi executado.** No PC do treino o Python ficou bloqueado. Ele pode ter bugs: se um teste falhar, corrija o código, explique o que mudou e rode de novo.

### 3. Gerar as métricas e as figuras (GPU opcional; na CPU leva ~10–20 min)

```
uv run python avaliar.py inferir principal          # métricas por imagem em runs/principal/metricas_teste.csv
uv run python avaliar.py casos principal            # casos_A2B.jpg e casos_B2A.jpg (melhores e piores pela cíclica L1)
uv run python avaliar.py esteganografia principal   # esteganografia.csv e .png (ruído σ 0,02 e 0,05)
uv run python analyze.py curvas runs/principal      # curvas_adversarial.png e curvas_ciclica.png
```

Atenção às diferenças de argumento: o `avaliar.py` recebe o **nome** da run (`principal`), e o `analyze.py` recebe o **caminho** (`runs/principal`). Com GPU, o `inferir` aceita `--fp16`.

Abra cada figura gerada e confira se faz sentido. Anote os números impressos: cíclica L1, PSNR e SSIM por sentido, e a tabela da esteganografia.

### 4. Completar os slides (não invente números; só use o que os scripts imprimirem)

- **`analise-estego`:** troque `Resultado: [a preencher]` pelo resultado do teste de ruído. Diga se a reconstrução piora muito mais do que o ruído explica. Se piorar, isso confirma a esteganografia; se não, diga que a hipótese não se confirma, que também é uma conclusão válida.
- **`limitacoes`:** troque `[a preencher]` pelas métricas no teste inteiro (cíclica L1, PSNR e SSIM, com o número de imagens).
- **Opcional:** um slide novo de métricas no teste com uma tabela pequena, entre `evolucao-b2a` e `analise-texto`, e/ou as figuras `casos_*.jpg`. Para isso, suba a imagem como asset do Artifact e edite `project/deck.json` (`order`).
- **`coleta`:** tem 3 placeholders entre colchetes sobre **como os dados foram coletados**. **Pergunte ao usuário**; só o grupo sabe.

### 5. Atualizar o README

Na seção "Resultados e análise" há 3 marcações `_pendente_`. Troque pelos números e figuras gerados. Inclua também a origem dos dados (TODO na seção "Origem dos dados") quando o usuário informar.

### 6. Commit e push

- Trabalhe na `main` (o grupo commita direto nela). Mensagens em português, no estilo `feat: ...` dos commits existentes.
- Adicione os arquivos um por um: os CSVs e as figuras novas em `runs/principal/`, mais o README e o código, se houve correção.
- **Nunca** commite `data/`, `dados.zip`, o PDF do enunciado nem outros `.pt` além do `G_epoca_100.pt`. O `.gitignore` já bloqueia esses arquivos; confira com `git status` antes.
- Existe também o subcomando `analyze.py traduzir`, uma demo que traduz imagens quaisquer com `--pesos runs/principal/G_epoca_100.pt` (veja o `COMO_RODAR.md`). Ele pode servir para uma simulação ao vivo na apresentação.

## Regras

- Não retreine e não rode o `train.py`.
- Não invente métricas nem descrições de imagens: tudo deve vir dos scripts ou de imagens que você abriu.
- Português nos slides, no README e nos commits.
- Resposta ao usuário curta e direta. Ele precisa entregar na quinta.
