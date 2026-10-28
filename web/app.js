/* Frontend estático do Churn Predictor — consome a API FastAPI.
 * Sem CDN, sem build, sem dependências externas.
 */
"use strict";

// ---- Configuração -------------------------------------------------------
const API_BASE = String(window.CHURN_API_BASE || "").replace(/\/+$/, "");
const isPagesHost = /(^|\.)github\.io$/i.test(location.hostname);
const isFileHost = location.protocol === "file:";

// Timeout padrão para todas as chamadas de API (R3).
const FETCH_TIMEOUT_MS = 15000;

// Campos do POST /predict (19 campos, nomes exatos do contrato).
const FIELD_NAMES = [
  "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
  "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
  "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
  "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
  "MonthlyCharges", "TotalCharges"
];
const INT_FIELD_NAMES = ["SeniorCitizen", "tenure"];
const NUM_FIELD_NAMES = ["MonthlyCharges", "TotalCharges"];

// ---- Helpers -------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function apiUrl(path) {
  return (API_BASE || "") + path;
}

function showBannerErro(msg) {
  const el = $("banner-erro");
  el.textContent = msg;
  el.hidden = !msg;
}

function setUltimoJson(label, data) {
  $("ultimo-json").textContent = "// " + label + "\n" + JSON.stringify(data, null, 2);
}

function setApiStatus(estado, texto) {
  const dot = $("api-dot");
  dot.classList.remove("dot--ok", "dot--warn", "dot--off");
  dot.classList.add("dot--" + estado);
  $("api-status-text").textContent = texto;
}

function fmtSegundos() {
  return Math.round(FETCH_TIMEOUT_MS / 1000);
}

/**
 * W4: converte valores primitivos para exibição. Aceita apenas string não
 * vazia, número finito ou booleano; qualquer outra coisa (objeto, array,
 * null, undefined, NaN, string vazia) vira o fallback "—".
 */
function valorPrimitivo(v, fallback) {
  const vazio = fallback === undefined ? "—" : fallback;
  if (typeof v === "string") return v === "" ? vazio : v;
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : vazio;
  if (typeof v === "boolean") return String(v);
  return vazio;
}

// ---- W4: marcação de acessibilidade dos campos inválidos -------------------
/** Marca o campo com aria-invalid e aponta aria-describedby para a mensagem. */
function marcarInvalido(campo, mensagem) {
  if (!campo || typeof campo.setAttribute !== "function") return;
  campo.setAttribute("aria-invalid", "true");
  const msgEl = campo.id ? $("erro-" + campo.id) : null;
  if (msgEl) {
    msgEl.textContent = mensagem;
    msgEl.hidden = false;
    campo.setAttribute("aria-describedby", msgEl.id);
  }
}

/** Remove a marcação de um único campo (usado ao corrigir/limpar). */
function limparInvalido(campo) {
  if (!campo || typeof campo.removeAttribute !== "function") return;
  campo.removeAttribute("aria-invalid");
  campo.removeAttribute("aria-describedby");
  const msgEl = campo.id ? $("erro-" + campo.id) : null;
  if (msgEl) {
    msgEl.textContent = "";
    msgEl.hidden = true;
  }
}

/** Remove a marcação de todos os campos validados. */
function limparTodasMarcas() {
  const form = $("form-predict");
  for (const nome of FIELD_NAMES) {
    limparInvalido(form.elements.namedItem(nome));
  }
}

/**
 * Busca JSON com timeout (AbortController, R3) e validação estrita (R1).
 * `validador(body)` recebe o objeto já parseado e deve retornar:
 *   - null/undefined  -> payload válido
 *   - string          -> descrição do problema (vira erro explícito)
 * Erros lançados carregam sinalização:
 *   err.timeout  -> tempo limite excedido
 *   err.invalid  -> HTTP ok, mas corpo não-JSON/JSON inválido/payload fora do contrato
 *   err.status   -> HTTP != 2xx
 * Sem `err.status` -> falha de rede (API offline).
 *   err.cancelled -> requisição invalidada por signal externo (W5)
 */
async function fetchJson(path, opts, validador) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  // W5: signal externo (cancelamento da predição anterior) também aborta a
  // requisição; o cancelamento é reportado separadamente do timeout.
  const externo = opts && opts.signal ? opts.signal : null;
  const aoCancelarExterno = () => controller.abort();
  if (externo) {
    if (externo.aborted) controller.abort();
    else externo.addEventListener("abort", aoCancelarExterno);
  }
  let res;
  let texto;
  try {
    res = await fetch(apiUrl(path), {
      method: (opts && opts.method) || "GET",
      headers: Object.assign({ "Accept": "application/json" }, (opts && opts.headers) || {}),
      body: opts && opts.body,
      signal: controller.signal
    });
    // Lê como texto para distinguir JSON válido de HTML/texto (R1).
    texto = await res.text();
  } catch (err) {
    if (externo && externo.aborted) {
      const c = new Error("requisição cancelada");
      c.cancelled = true;
      throw c;
    }
    if (controller.signal.aborted) {
      const t = new Error("A API não respondeu dentro do tempo limite de " + fmtSegundos() + " s.");
      t.timeout = true;
      throw t;
    }
    throw err; // erro de rede
  } finally {
    clearTimeout(timer);
    if (externo) externo.removeEventListener("abort", aoCancelarExterno);
  }

  let body = null;
  let parseOk = true;
  try {
    body = JSON.parse(texto);
  } catch (_) {
    parseOk = false;
  }

  if (!res.ok) {
    const e = new Error("HTTP " + res.status);
    e.status = res.status;
    e.body = parseOk ? body : { _texto: String(texto || "").slice(0, 200) };
    throw e;
  }

  // HTTP 200 que não é um objeto JSON não pode ser tratado como sucesso (R1).
  if (!parseOk || body === null || typeof body !== "object" || Array.isArray(body)) {
    const e = new Error("a resposta não é um objeto JSON válido");
    e.status = res.status;
    e.invalid = true;
    e.rawText = String(texto || "");
    throw e;
  }

  if (typeof validador === "function") {
    const problema = validador(body);
    if (problema) {
      const e = new Error(problema);
      e.status = res.status;
      e.invalid = true;
      e.rawText = String(texto || "");
      throw e;
    }
  }
  return body;
}

// ---- Validadores de payload (R1) ----------------------------------------
function validarHealth(h) {
  if (typeof h.status !== "string" || h.status === "") return "payload de /health sem o campo obrigatório \"status\"";
  if (typeof h.model_loaded !== "boolean") return "payload de /health sem o campo obrigatório \"model_loaded\" (booleano)";
  return null;
}

function validarModelInfo(mi) {
  if (typeof mi.model_type !== "string" || mi.model_type === "") return "payload de /model-info sem \"model_type\"";
  if (typeof mi.model_version !== "string" || mi.model_version === "") return "payload de /model-info sem \"model_version\"";
  if (!mi.metrics || typeof mi.metrics !== "object" || Array.isArray(mi.metrics)) return "payload de /model-info sem \"metrics\"";
  // W1: feature_importances, quando presente, deve ser lista de objetos
  // { feature: string não vazia, importance: número finito }. Itens fora do
  // contrato viram erro explícito de contrato — nunca TypeError nem
  // "API offline" para uma resposta HTTP 200.
  if (mi.feature_importances !== undefined && mi.feature_importances !== null) {
    if (!Array.isArray(mi.feature_importances)) {
      return "payload de /model-info com \"feature_importances\" fora do contrato (esperado lista)";
    }
    for (let i = 0; i < mi.feature_importances.length; i++) {
      const item = mi.feature_importances[i];
      if (!item || typeof item !== "object" || Array.isArray(item) ||
          typeof item.feature !== "string" || item.feature === "" ||
          typeof item.importance !== "number" || !Number.isFinite(item.importance)) {
        return "item inválido em \"feature_importances\" (posição " + i +
          "; esperado objeto com \"feature\" (texto) e \"importance\" (número))";
      }
    }
  }
  return null;
}

function validarPredict(r) {
  const p = r.churn_probability;
  if (typeof p !== "number" || !Number.isFinite(p) || p < 0 || p > 1) {
    return "resposta de /predict sem \"churn_probability\" numérica válida em [0,1]";
  }
  if (typeof r.churn_prediction !== "boolean") {
    return "resposta de /predict sem \"churn_prediction\" booleana";
  }
  return null;
}

// ---- Painel Backend ------------------------------------------------------
function configurarLinksBackend() {
  $("backend-base-url").textContent = API_BASE || location.origin + " (mesma origem)";
  $("link-docs").href = apiUrl("/docs");
  $("link-health").href = apiUrl("/health");
  $("link-model-info").href = apiUrl("/model-info");
}

// ---- /health + /model-info ----------------------------------------------
async function carregarHealth() {
  try {
    const h = await fetchJson("/health", null, validarHealth);
    setUltimoJson("GET /health", h);
    if (h.status === "ok" && h.model_loaded) {
      setApiStatus("ok", "API online — modelo " + (h.model_version || "?"));
    } else {
      setApiStatus("warn", "API respondeu, mas o modelo não está carregado.");
    }
    return h;
  } catch (err) {
    let motivo;
    if (err.timeout) motivo = "tempo limite de " + fmtSegundos() + " s excedido";
    else if (err.invalid) motivo = "resposta HTTP " + err.status + " com conteúdo não-JSON";
    else if (err.status) motivo = "resposta HTTP " + err.status;
    else motivo = "sem conexão";
    setApiStatus("off", "API indisponível: " + motivo + ".");
    return null;
  }
}

function renderMetricas(metrics) {
  const nomes = { accuracy: "Accuracy", precision: "Precision", recall: "Recall", f1: "F1", roc_auc: "ROC AUC" };
  const dl = $("model-metricas");
  dl.textContent = "";
  for (const key of Object.keys(nomes)) {
    const wrap = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = nomes[key];
    const dd = document.createElement("dd");
    // Métrica ausente/nula mostra "—", nunca 0.000 (R6).
    const v = metrics ? metrics[key] : undefined;
    const num = typeof v === "number" ? v : NaN;
    dd.textContent = Number.isFinite(num) ? num.toFixed(3) : "—";
    wrap.append(dt, dd);
    dl.append(wrap);
  }
}

function renderFeatures(list) {
  const box = $("feature-bars");
  box.textContent = "";
  if (!Array.isArray(list) || list.length === 0) return;
  // W1: defende contra itens fora do contrato (null, sem feature/importance) —
  // nunca deixar um TypeError escapar para o catch "API offline".
  const validas = list.filter(
    (f) => f && typeof f === "object" &&
      typeof f.feature === "string" && f.feature !== "" &&
      typeof f.importance === "number" && Number.isFinite(f.importance)
  );
  if (validas.length === 0) return;
  const max = Math.max(...validas.map((f) => f.importance)) || 1;
  const ordenadas = validas
    .map((f) => ({ feature: f.feature, importance: f.importance }))
    .sort((a, b) => b.importance - a.importance);
  for (const f of ordenadas) {
    const row = document.createElement("div");
    row.className = "feature-row";
    const rotulo = document.createElement("span");
    rotulo.className = "feature-nome";
    rotulo.textContent = f.feature;
    const track = document.createElement("div");
    track.className = "feature-track";
    track.setAttribute("role", "img");
    track.setAttribute("aria-label", f.feature + ": importância " + f.importance.toFixed(4));
    const fill = document.createElement("span");
    fill.className = "feature-fill";
    fill.style.width = Math.max(2, (f.importance / max) * 100) + "%";
    track.append(fill);
    const valor = document.createElement("span");
    valor.className = "feature-valor";
    valor.textContent = f.importance.toFixed(3);
    row.append(rotulo, track, valor);
    box.append(row);
  }
}

async function carregarModelInfo() {
  try {
    const mi = await fetchJson("/model-info", null, validarModelInfo);
    setUltimoJson("GET /model-info", mi);
    const chips = [
      "Modelo: " + valorPrimitivo(mi.model_type, "?"),
      "Versão: " + valorPrimitivo(mi.model_version, "?"),
      // W4: só exibe primitivos válidos; objeto/array/null viram fallback.
      "Features: " + valorPrimitivo(mi.n_features, "?"),
      "Estimators: " + valorPrimitivo(mi.n_estimators, "?"),
      "Threshold: " + valorPrimitivo(mi.decision_threshold, "?")
    ];
    const ul = $("model-meta");
    ul.textContent = "";
    for (const c of chips) {
      const li = document.createElement("li");
      li.textContent = c;
      ul.append(li);
    }
    renderMetricas(mi.metrics);
    renderFeatures(mi.feature_importances || []);
    $("model-loading").hidden = true;
    $("model-info").hidden = false;
    return mi;
  } catch (err) {
    // Payload inválido mostra mensagem de erro, não um painel vazio (R1).
    let msg;
    if (err.invalid) {
      msg = "Resposta inválida de /model-info (HTTP " + err.status + "): " + err.message +
        ". Verifique se CHURN_API_BASE aponta para o backend FastAPI correto.";
    } else if (err.timeout) {
      // W3: err.message já descreve o tempo limite; não duplicar o texto.
      msg = err.message + " (=/model-info)";
    } else if (err.status) {
      msg = "Não foi possível carregar /model-info (HTTP " + err.status + ").";
    } else {
      msg = "Não foi possível carregar /model-info (API offline).";
    }
    $("model-loading").hidden = false;
    $("model-loading").textContent = msg + " O painel aparece quando o backend estiver acessível.";
    $("model-info").hidden = true;
    return null;
  }
}

// ---- Formulário ----------------------------------------------------------
/**
 * Lê o formulário como TEXTO BRUTO e valida antes da conversão (R2/R6):
 * - campos numéricos vazios são rejeitados (nunca viram 0 silenciosamente);
 * - tenure deve ser inteiro (rejeita "12.7"), 0–72;
 * - MonthlyCharges 0–200; TotalCharges >= 0; SeniorCitizen 0/1.
 * Retorna { dados, problemas, invalidos, primeiroInvalido } — `invalidos`
 * traz { campo, mensagem } para a marcação de acessibilidade (W4).
 */
function coletarEValidar() {
  const form = $("form-predict");
  const dados = {};
  const problemas = [];
  const invalidos = [];
  let primeiroInvalido = null;

  function registrarProblema(campo, mensagem) {
    problemas.push(mensagem);
    invalidos.push({ campo, mensagem });
    if (!primeiroInvalido) primeiroInvalido = campo;
  }

  for (const nome of FIELD_NAMES) {
    const campo = form.elements.namedItem(nome);
    if (!campo) {
      return { dados: null, problemas: ["Campo ausente no formulário: " + nome], invalidos: [], primeiroInvalido: null };
    }
    const bruto = String(campo.value == null ? "" : campo.value).trim();

    if (nome === "tenure") {
      if (bruto === "") {
        registrarProblema(campo, "tenure está vazio.");
      } else if (!/^\d+$/.test(bruto)) {
        // W4: quando havia texto, não dizer "está vazio"; dizer o que falta.
        registrarProblema(campo, /\./.test(bruto)
          ? 'tenure "' + bruto + '" tem casas decimais; use um número inteiro de 0 a 72.'
          : 'tenure "' + bruto + '" não é um número inteiro; use um valor de 0 a 72.');
      } else if (Number(bruto) > 72) {
        registrarProblema(campo, "tenure deve estar entre 0 e 72.");
      } else {
        dados.tenure = Number(bruto);
        continue;
      }
    } else if (nome === "SeniorCitizen") {
      if (bruto !== "0" && bruto !== "1") {
        registrarProblema(campo, "SeniorCitizen deve ser 0 ou 1.");
      } else {
        dados.SeniorCitizen = Number(bruto);
      }
    } else if (NUM_FIELD_NAMES.includes(nome)) {
      const max = nome === "MonthlyCharges" ? 200 : Infinity;
      const regra = nome === "MonthlyCharges"
        ? "MonthlyCharges deve estar entre 0 e 200."
        : "TotalCharges deve ser um número >= 0.";
      if (bruto === "") {
        registrarProblema(campo, nome + " está vazio.");
      } else if (!Number.isFinite(Number(bruto)) || Number(bruto) < 0 || Number(bruto) > max) {
        registrarProblema(campo, regra);
      } else {
        dados[nome] = Number(bruto);
        continue;
      }
    } else {
      if (bruto === "") {
        registrarProblema(campo, nome + " está vazio.");
      } else {
        dados[nome] = bruto;
      }
    }
  }

  return { dados: problemas.length ? null : dados, problemas, invalidos, primeiroInvalido };
}

function detalharErro422(body) {
  // W3: nunca lançar dentro do catch — itens nulos ou fora do formato
  // viram o fallback, e qualquer exceção inesperada é absorvida.
  try {
    if (body && Array.isArray(body.detail)) {
      const partes = body.detail.map((d) => {
        const loc = d && Array.isArray(d.loc) && d.loc.length ? d.loc.slice(1).join(".") || "?" : "?";
        const msg = d && typeof d.msg === "string" && d.msg !== "" ? d.msg : "valor inválido";
        return loc + ": " + msg;
      });
      return partes.join(" | ");
    }
    if (body && typeof body.detail === "string") return body.detail;
  } catch (_) {
    // Silencioso de propósito: a chamada já falha com HTTP != 2xx.
  }
  return "resposta inválida do servidor";
}

function riscoClasse(nivel) {
  return { baixo: "risco--baixo", medio: "risco--medio", alto: "risco--alto" }[nivel] || "";
}

// ---- Predição -------------------------------------------------------------
// W5: identificador sequencial do pedido de /predict em andamento + seu
// AbortController. "Preencher exemplo"/reset incrementa o contador e cancela
// a requisição, de modo que nenhum resultado antigo volte a aparecer na tela.
let sequenciaPredict = 0;
let controllerPredict = null;

async function prever(evento) {
  evento.preventDefault();
  const btn = $("btn-prever");
  const erroEl = $("erro-form");
  erroEl.hidden = true;
  $("resultado").hidden = true;

  const coleta = coletarEValidar();
  limparTodasMarcas();
  if (!coleta.dados) {
    erroEl.textContent = coleta.problemas.length === 1
      ? coleta.problemas[0]
      : "Corrija os valores: " + coleta.problemas.join(" ");
    erroEl.hidden = false;
    // W4: marca cada campo inválido (aria-invalid + aria-describedby).
    for (const item of coleta.invalidos) marcarInvalido(item.campo, item.mensagem);
    if (coleta.primeiroInvalido) coleta.primeiroInvalido.focus();
    return;
  }
  const dados = coleta.dados;

  btn.disabled = true;
  btn.classList.add("btn--carregando");
  btn.setAttribute("aria-busy", "true");
  $("pred-loading").hidden = false;
  showBannerErro("");

  const meuPedido = ++sequenciaPredict;
  controllerPredict = new AbortController();

  try {
    const r = await fetchJson(
      "/predict",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dados),
        signal: controllerPredict.signal
      },
      validarPredict
    );
    // W5: se outro pedido/reset assumiu o controle, descarta este resultado.
    if (meuPedido !== sequenciaPredict) return;
    // Só chega aqui se churn_probability ∈ [0,1] for número finito e
    // churn_prediction for booleano (R1).
    setUltimoJson("POST /predict", r);
    const prob = r.churn_probability;
    $("res-prob").textContent = prob.toFixed(4);
    $("res-classe").textContent = r.churn_prediction ? "Vai cancelar (churn = 1)" : "Não vai cancelar (churn = 0)";
    const risco = $("res-risco");
    risco.textContent = typeof r.risk_level === "string" ? r.risk_level : "—";
    risco.className = riscoClasse(r.risk_level);
    // W4: decision_threshold e model_version só aparecem se forem primitivos.
    $("res-threshold").textContent = valorPrimitivo(r.decision_threshold);
    $("res-versao").textContent = valorPrimitivo(r.model_version);
    const fill = $("barra-prob-fill");
    fill.style.width = Math.min(100, Math.max(0, prob * 100)) + "%";
    fill.className = riscoClasse(r.risk_level);
    $("resultado").hidden = false;
    setApiStatus("ok", "API online — predição concluída.");
  } catch (err) {
    // W5: cancelamento (reset/novo pedido) não deve mexer na UI.
    if (err.cancelled || meuPedido !== sequenciaPredict) return;
    if (err.timeout) {
      // W3: err.message já traz o tempo limite; não duplicar o texto.
      erroEl.textContent = err.message;
      erroEl.hidden = false;
      setApiStatus("off", "API sem resposta dentro do tempo limite.");
    } else if (err.invalid) {
      // 200 com HTML/não-JSON/payload fora do contrato: erro explícito (R1).
      setUltimoJson("POST /predict respondeu conteúdo inválido", {
        http_status: err.status,
        trecho_da_resposta: err.rawText.slice(0, 200)
      });
      erroEl.textContent = "A API respondeu (HTTP " + err.status + "), mas com conteúdo inválido: " +
        err.message + ". Verifique se CHURN_API_BASE aponta para a API FastAPI (não para o próprio site).";
      erroEl.hidden = false;
      setApiStatus("off", "API respondeu com conteúdo inesperado.");
    } else if (err.status) {
      const msg = "A API retornou HTTP " + err.status + " — " + detalharErro422(err.body);
      setUltimoJson("POST /predict falhou", err.body);
      erroEl.textContent = msg;
      erroEl.hidden = false;
    } else {
      erroEl.hidden = true;
      showBannerErro(
        "API offline: não foi possível alcançar " +
        (API_BASE || location.origin) +
        ". Suba o backend (docker compose up ou uvicorn churn.api:app) e tente novamente."
      );
      setApiStatus("off", "API offline.");
    }
  } finally {
    // R3: botão e "Calculando…" sempre são liberados (sucesso, erro,
    // timeout ou cancelamento).
    btn.disabled = false;
    btn.classList.remove("btn--carregando");
    btn.removeAttribute("aria-busy");
    $("pred-loading").hidden = true;
    if (controllerPredict && meuPedido === sequenciaPredict) controllerPredict = null;
  }
}

/**
 * W5: invalida um /predict em andamento (chamado pelo reset/"Preencher
 * exemplo"). O contador impede qualquer renderização tardia e o abort
 * encerra a requisição sem deixar "resultado fantasma" na página.
 */
function invalidarPredictEmAndamento() {
  sequenciaPredict++;
  if (controllerPredict) {
    try {
      controllerPredict.abort();
    } catch (_) {
      // Abort nunca deve quebrar o reset.
    }
    controllerPredict = null;
  }
}

// ---- Dependência de campos (UX, sem alterar o contrato) ------------------
// PhoneService = No fixa MultipleLines em "No phone service" e InternetService
// = No fixa os add-ons em "No internet service"; os selects são desabilitados
// porque esses são os únicos valores aceitos pelo contrato da API para esses
// casos (evita enviar combinação inválida).
function sincronizarDependencias() {
  const form = $("form-predict");
  const phone = form.elements.PhoneService;
  const lines = form.elements.MultipleLines;
  const internet = form.elements.InternetService;
  const internetAddons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"];

  if (phone.value === "No") {
    lines.value = "No phone service";
    lines.disabled = true;
  } else {
    lines.disabled = false;
    if (lines.value === "No phone service") lines.value = "No";
  }
  for (const nome of internetAddons) {
    const campo = form.elements[nome];
    if (internet.value === "No") {
      campo.value = "No internet service";
      campo.disabled = true;
    } else {
      campo.disabled = false;
      if (campo.value === "No internet service") campo.value = "No";
    }
  }
}

// ---- Avisos de configuração (R6) ------------------------------------------
// Mensagens distintas: GitHub Pages (github.io) vs. arquivos abertos via file://.
// Em file:// não se fala em Pages; a mensagem é neutra ("API não configurada").
function verificarConfigPages(healthOk) {
  const aviso = $("pages-aviso");
  const avisoPages = $("aviso-pages");
  const avisoFile = $("aviso-file");
  if (API_BASE || healthOk) {
    aviso.hidden = true;
    return;
  }
  if (isPagesHost) {
    avisoPages.hidden = false;
    avisoFile.hidden = true;
    aviso.hidden = false;
  } else if (isFileHost) {
    avisoFile.hidden = false;
    avisoPages.hidden = true;
    aviso.hidden = false;
  } else {
    aviso.hidden = true;
  }
}

// ---- Inicialização ---------------------------------------------------------
async function init() {
  configurarLinksBackend();
  $("form-predict").addEventListener("submit", prever);
  $("btn-limpar").addEventListener("click", () => {
    // W5: invalida um /predict em andamento antes de limpar, para que o
    // resultado antigo não reapareça quando a resposta chegar.
    invalidarPredictEmAndamento();
    $("form-predict").reset();
    sincronizarDependencias();
    $("erro-form").hidden = true;
    // R6: reset precisa ocultar o resultado da predição anterior.
    $("resultado").hidden = true;
    // W4: limpa as marcações de campo inválido.
    limparTodasMarcas();
  });
  // W4: ao corrigir um campo, remove aria-invalid/aria-describedby e a
  // mensagem associada.
  $("form-predict").addEventListener("input", (ev) => {
    if (ev.target && ev.target.getAttribute && ev.target.getAttribute("aria-invalid") === "true") {
      limparInvalido(ev.target);
    }
  });
  $("form-predict").addEventListener("change", (ev) => {
    if (ev.target && ev.target.getAttribute && ev.target.getAttribute("aria-invalid") === "true") {
      limparInvalido(ev.target);
    }
  });
  $("form-predict").elements.PhoneService.addEventListener("change", sincronizarDependencias);
  $("form-predict").elements.InternetService.addEventListener("change", sincronizarDependencias);
  sincronizarDependencias();

  const [health] = await Promise.allSettled([carregarHealth(), carregarModelInfo()]);
  verificarConfigPages(health.status === "fulfilled" && health.value !== null);
}

document.addEventListener("DOMContentLoaded", init);
