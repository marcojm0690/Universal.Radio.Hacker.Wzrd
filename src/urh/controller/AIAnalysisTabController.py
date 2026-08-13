import json
import urllib.error
import urllib.request

from PyQt6.QtCore import pyqtSlot, pyqtSignal, QThread, Qt
from PyQt6.QtWidgets import QWidget, QMessageBox

from urh import settings
from urh.controller.CompareFrameController import CompareFrameController
from urh.signalprocessing.MessageType import MessageType
from urh.signalprocessing.ProtocolAnalyzer import ProtocolAnalyzer
from urh.ui.ui_ai_analysis import Ui_AIAnalysisTab
from urh.util.Logger import logger


LOADED_SUFFIX = " (loaded)"
MAX_RESPONSE_TOKENS = 4096


class LMStudioWorker(QThread):
    """
    Runs the request to the LM Studio server (OpenAI compatible API) in a
    background thread so the GUI does not freeze while the model is thinking.
    """

    finished_with_result = pyqtSignal(str, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        server_url: str,
        model: str,
        messages: list,
        temperature=0.2,
        max_tokens=MAX_RESPONSE_TOKENS,
        parent=None,
    ):
        super().__init__(parent)
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self):
        try:
            logger.info(
                "Requesting analysis from model '{0}' at {1}".format(
                    self.model, self.server_url
                )
            )
            payload = json.dumps(
                {
                    "model": self.model,
                    "messages": self.messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                }
            ).encode("utf-8")

            request = urllib.request.Request(
                self.server_url + "/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))

            message = result["choices"][0]["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            finish_reason = result["choices"][0].get("finish_reason", "")
            model_used = result.get("model", self.model)

            if not content:
                # Reasoning models put the thinking into "reasoning_content" and
                # leave "content" empty until the chain of thought is complete.
                if reasoning:
                    content = (
                        "[Thinking]\n{0}\n\n[Answer]\n(no final answer was produced)".format(
                            reasoning
                        )
                    )
                elif finish_reason == "length":
                    content = (
                        "(The model ran out of output tokens before producing an answer. "
                        "Try a smaller capture or a model with a larger context.)"
                    )
                else:
                    content = "(The model returned an empty response.)"

            self.finished_with_result.emit(model_used, content)
            logger.info(
                "Analysis result from '{0}' (finish_reason={1}, content_len={2}, reasoning_len={3})".format(
                    model_used, finish_reason, len(content), len(reasoning)
                )
            )
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
            except Exception:
                pass
            self.failed.emit("{} ({}): {}".format(str(e), e.code, detail or "model not available in LM Studio"))
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            self.failed.emit(str(e))
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.failed.emit("Unexpected response from server: {}".format(e))
        except Exception as e:
            self.failed.emit("{}: {}".format(type(e).__name__, e))


class AIAnalysisTabController(QWidget):
    DEFAULT_SERVER = "http://127.0.0.1:1234"

    def __init__(self, compare_frame_controller: CompareFrameController, parent=None):
        super().__init__(parent)

        self.compare_frame_controller = compare_frame_controller
        self.ui = Ui_AIAnalysisTab()
        self.ui.setupUi(self)

        self.__worker = None
        self.__context_lengths = {}
        self.last_truncation_note = ""

        self.captured_data_text = ""
        self.prompt_template = ""

        self.restore_settings()
        self.create_connects()
        self.refresh_captured_data()

    def restore_settings(self):
        server = settings.read("ai_server", self.DEFAULT_SERVER, str)
        if not server:
            server = self.DEFAULT_SERVER
        self.ui.lnEdtServer.setText(server)

        model = settings.read("ai_model", "", str)
        if model:
            if model.endswith(LOADED_SUFFIX):
                model = model[: -len(LOADED_SUFFIX)]
            self.ui.cbModel.setEditable(True)
            self.ui.cbModel.setCurrentText(model)

    def create_connects(self):
        self.ui.btnRefreshModels.clicked.connect(self.refresh_models)
        self.ui.btnConnect.clicked.connect(self.check_connection)
        self.ui.btnRefreshData.clicked.connect(self.refresh_captured_data)
        self.ui.btnAnalyze.clicked.connect(self.analyze)
        self.ui.btnAnalyzeDeep.clicked.connect(self.analyze_deep)
        self.ui.btnStop.clicked.connect(self.stop_worker)
        self.ui.btnClear.clicked.connect(self.ui.txtResult.clear)
        self.ui.cbModel.editTextChanged.connect(self.on_model_changed)

    def on_model_changed(self, text: str):
        clean = text[: -len(LOADED_SUFFIX)] if text.endswith(LOADED_SUFFIX) else text
        settings.write("ai_model", clean.strip())

    @property
    def server_url(self) -> str:
        return self.ui.lnEdtServer.text().strip() or self.DEFAULT_SERVER

    @property
    def selected_model(self) -> str:
        model = ""
        if self.ui.cbModel.currentData():
            model = str(self.ui.cbModel.currentData())
        else:
            model = self.ui.cbModel.currentText().strip()
        if model.endswith(LOADED_SUFFIX):
            model = model[: -len(LOADED_SUFFIX)]
        return model

    def __set_status(self, text: str, error=False):
        self.ui.lblStatus.setText(text)
        if error:
            self.ui.lblStatus.setStyleSheet("color: #d44;")
        else:
            self.ui.lblStatus.setStyleSheet("")

    def __build_payload(self, mode: str) -> list:
        """
        Build the chat messages for the LLM from the captured protocol data.

        :param mode: "quick" or "deep"
        """
        system_prompt = (
            "You are an expert in wireless protocol reverse engineering and SDR signal analysis. "
            "You analyze captured radio protocol data (bits, hex, fields, modulation metadata) that was "
            "recorded and demodulated with the Universal Radio Hacker (URH). "
            "Be concise and structured. Use short bullet points, not long paragraphs. "
            "When you guess a device or protocol, always state your confidence (low/medium/high) and the "
            "reasoning in one short line. Base your analysis only on the data provided and on well-known "
            "wireless protocol knowledge."
        )

        user_prompt = self.prompt_template
        if mode == "deep":
            user_prompt += (
                "\n\nAdditionally, give concrete engineering advice to improve the reverse engineering "
                "of this protocol. Keep it short:\n"
                "1. Which additional captures/measurements would help (more samples, different states, "
                "rolling code check)?\n"
                "2. Which demodulation or decoding steps to try next in URH (modulation type, data "
                "whitening, bit order, Manchester, differential)?\n"
                "3. How to identify or crack the checksum algorithm if present.\n"
                "4. How to confirm or refute your device/protocol guess (specific fields to verify).\n"
                "5. Potential attack/analysis surface (replay, fuzzing, stateful simulation) and caveats.\n"
            )

        user_prompt += "\n\n"

        data_section = (
            "===== CAPTURED DATA =====\n"
            + (self.captured_data_text or "(no protocol data loaded yet)")
            + "\n===== END OF DATA =====\n\n"
            "Based on the data above, provide your analysis."
        )

        # Estimate token usage: prompt template + wrapped data section.
        # Conservative heuristic: ~2.5 chars per token (hex/bits compress well).
        chars_per_token = 2.5
        context = self.__context_lengths.get(self.selected_model, 8192)
        prompt_budget = max(512, context - MAX_RESPONSE_TOKENS - 200)
        budget_chars = int(prompt_budget * chars_per_token)

        self.last_truncation_note = ""
        data_chars = len(data_section)
        base_chars = len(user_prompt)
        if base_chars + data_chars > budget_chars:
            data_budget = max(0, budget_chars - base_chars)
            data_section = self.__truncate_data_section(data_section, data_budget)
            self.last_truncation_note = (
                "Captured data was truncated to fit the model's {0}-token context "
                "window. Consider loading a model with a larger context for full captures.".format(
                    context
                )
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + data_section},
        ]

    @staticmethod
    def __truncate_data_section(data_section: str, budget_chars: int) -> str:
        if len(data_section) <= budget_chars:
            return data_section

        header = "===== CAPTURED DATA ====="
        header_end = len(header) + 1  # past the header line

        # Prefer cutting on a message boundary, but fall back to a mid-line cut
        # so a single long line still yields a large, useful payload.
        cut = data_section.rfind("\n", header_end, budget_chars)
        if cut < header_end:
            cut = budget_chars

        truncated = data_section[:cut].rstrip()
        omitted = len(data_section) - len(truncated)
        truncated += "\n\n[... truncated: {0} chars of captured data omitted to fit the model's "
        truncated += "context window; refresh and re-analyze with a smaller capture if needed ...]"
        return truncated.format(omitted)

    @pyqtSlot()
    def refresh_captured_data(self):
        self.captured_data_text = self.build_captured_data_text()
        self.ui.txtCapturedData.setPlainText(self.captured_data_text)
        self.ui.lblStatus.setText("Captured data refreshed")

    @pyqtSlot()
    def refresh_models(self):
        self.__set_status("Fetching models...")
        try:
            loaded_models = set()
            self.__context_lengths = {}
            try:
                url = self.server_url + "/api/v0/models"
                with urllib.request.urlopen(url, timeout=10) as response:
                    result = json.loads(response.read().decode("utf-8"))
                for m in result.get("data", []):
                    if m.get("state") == "loaded":
                        loaded_models.add(m["id"])
                    if m.get("loaded_context_length"):
                        self.__context_lengths[m["id"]] = m["loaded_context_length"]
            except Exception:
                pass

            url = self.server_url + "/v1/models"
            with urllib.request.urlopen(url, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))

            models = [m["id"] for m in result.get("data", [])]
            self.ui.cbModel.clear()
            self.ui.cbModel.setEditable(True)
            for model in sorted(models):
                if model in loaded_models:
                    self.ui.cbModel.addItem(model + LOADED_SUFFIX, userData=model)
                else:
                    self.ui.cbModel.addItem(model, userData=model)

            if loaded_models:
                target = next(iter(sorted(loaded_models)))
                index = self.ui.cbModel.findData(target)
                if index >= 0:
                    self.ui.cbModel.setCurrentIndex(index)
                else:
                    self.ui.cbModel.setCurrentText(target)

            if models:
                self.__set_status("{} model(s) available, {} loaded".format(len(models), len(loaded_models)))
            else:
                self.__set_status("No models available", error=True)
        except Exception as e:
            self.__set_status("Could not fetch models", error=True)
            QMessageBox.critical(
                self,
                self.tr("Connection error"),
                self.tr("Could not reach LM Studio at {0}:\n{1}").format(self.server_url, e),
            )

    @pyqtSlot()
    def check_connection(self):
        self.__set_status("Testing connection...")
        try:
            with urllib.request.urlopen(self.server_url + "/v1/models", timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
            num_models = len(result.get("data", []))
            self.__set_status("Connected ({0} model(s) available)".format(num_models))
        except Exception as e:
            self.__set_status("Connection failed", error=True)
            QMessageBox.critical(
                self,
                self.tr("Connection error"),
                self.tr("Could not reach LM Studio at {0}:\n{1}\n\n"
                        "Make sure the LM Studio local server is running (Developer tab -> "
                        "Start Server).").format(self.server_url, e),
            )

    def analyze(self):
        self.__start_analysis("quick")

    def analyze_deep(self):
        self.__start_analysis("deep")

    def __start_analysis(self, mode: str):
        if not self.captured_data_text:
            self.refresh_captured_data()

        if not self.selected_model:
            QMessageBox.warning(
                self,
                self.tr("No model selected"),
                self.tr("Please select a model. Use 'Refresh Models' to list the models "
                        "loaded in LM Studio."),
            )
            return

        if self.__worker is not None and self.__worker.isRunning():
            return

        messages = self.__build_payload(mode)
        context = self.__context_lengths.get(self.selected_model, 8192)
        placeholder = "Asking {} ...\n\nThis can take a while for local models.".format(
            self.selected_model
        )
        if self.last_truncation_note:
            placeholder = "Asking {} ... (context {})\n\n{}\n".format(
                self.selected_model, context, self.last_truncation_note
            )
        self.ui.txtResult.setPlainText(placeholder)
        self.__set_status("Analyzing...")

        self.__worker = LMStudioWorker(
            self.server_url,
            self.selected_model,
            messages,
            parent=self,
        )
        self.__worker.finished_with_result.connect(self.__on_analysis_finished)
        self.__worker.failed.connect(self.__on_analysis_failed)
        self.__worker.start()

    def stop_worker(self):
        if self.__worker is not None and self.__worker.isRunning():
            self.__worker.requestInterruption()
            self.__worker.terminate()
            self.__worker.wait(1000)
            self.__set_status("Stopped")

    @pyqtSlot(str, str)
    def __on_analysis_finished(self, model_used: str, content: str):
        self.ui.txtResult.setHtml(self.__render_markdownish(content))
        self.__set_status("Done ({})".format(model_used))
        settings.write("ai_model", self.selected_model)
        settings.write("ai_server", self.server_url)

    @pyqtSlot(str)
    def __on_analysis_failed(self, error: str):
        self.__set_status("Analysis failed", error=True)
        self.ui.txtResult.setPlainText(
            "Analysis failed:\n{}\n\nMake sure LM Studio is running and the server is "
            "started (Developer tab).".format(error)
        )

    @staticmethod
    def __render_markdownish(text: str) -> str:
        """
        Minimal markdown -> HTML converter good enough for LLM output.
        Handles code fences, inline code, headers, bold and lists.
        """
        import html
        import re

        escaped = html.escape(text)

        # Code fences
        def fence_repl(match):
            return "<pre style='background:#2b2b2b;color:#e0e0e0;padding:8px;border-radius:4px;'>" + match.group(2) + "</pre>"

        escaped = re.sub(r"```(\w*)\n?(.*?)```", fence_repl, escaped, flags=re.DOTALL)

        # Headers
        for level, size in [(3, "18px"), (2, "16px"), (1, "14px")]:
            escaped = re.sub(
                r"^#{%d}\s+(.+)$" % level,
                r"<h%d style='margin:8px 0;'>\1</h%d>" % (level, level),
                escaped,
                flags=re.MULTILINE,
            )

        escaped = re.sub(r"^\s*[-*]\s+(.+)$", r"<li>\1</li>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"<li>([^<]+)</li>", r"<ul><li>\1</li></ul>", escaped)
        escaped = re.sub(r"</ul><ul>", "", escaped)

        escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)

        escaped = re.sub(r"\n{2,}", "</p><p>", escaped)
        escaped = re.sub(r"\n", "<br>", escaped)

        return "<p style='font-size:13px;'>" + escaped + "</p>"

    def build_captured_data_text(self) -> str:
        """
        Serialize all currently loaded protocols (and their signals) into a
        text representation suitable to feed an LLM.
        """
        protocols = self.compare_frame_controller.full_protocol_list
        if not protocols:
            return (
                "No protocol data loaded.\n\n"
                "Open or capture a signal first (File -> Open or Record), or load a "
                "protocol file. Then press 'Refresh Captured Data'."
            )

        lines = []
        lines.append("WIRELESS PROTOCOL CAPTURE ANALYSIS")
        lines.append("=" * 50)
        lines.append("Number of protocols: {}".format(len(protocols)))

        for protocol in protocols:
            lines.extend(self.__protocol_to_text(protocol))

        return "\n".join(lines)

    def __protocol_to_text(self, protocol: ProtocolAnalyzer) -> list:
        lines = []
        lines.append("")
        lines.append("-" * 50)
        lines.append("PROTOCOL: {}".format(protocol.name))

        signal = protocol.signal
        if signal is not None:
            lines.append("Signal metadata:")
            lines.append("  - Sample rate: {} Hz".format(signal.sample_rate))
            lines.append("  - Modulation type: {}".format(signal.modulation_type))
            lines.append("  - Samples per symbol: {}".format(signal.samples_per_symbol))
            lines.append("  - Noise threshold: {}".format(signal.noise_threshold))
            lines.append("  - Center: {}".format(signal.center))
            lines.append("  - Center spacing: {}".format(signal.center_spacing))
            lines.append("  - Bits per symbol: {}".format(signal.bits_per_symbol))
            try:
                freq_one = protocol.estimate_frequency_for_one(signal.sample_rate)
                freq_zero = protocol.estimate_frequency_for_zero(signal.sample_rate)
                lines.append("  - Est. frequency (1-bit): {} Hz".format(freq_one))
                lines.append("  - Est. frequency (0-bit): {} Hz".format(freq_zero))
            except Exception:
                pass

        lines.append("Message types:")
        for message_type in protocol.message_types:
            lines.append("  - {}".format(self.__message_type_to_text(message_type)))

        lines.append("Messages:")
        for i, message in enumerate(protocol.messages):
            lines.append(self.__message_to_text(i, message))

        return lines

    @staticmethod
    def __message_type_to_text(message_type: MessageType) -> str:
        if len(message_type) == 0:
            return "{}: (no fields)".format(message_type.name)

        parts = []
        for label in message_type:
            parts.append(
                "{}[{}:{}]".format(label.name, label.start, label.end)
            )
        return "{}: {}".format(message_type.name, ", ".join(parts))

    @staticmethod
    def __message_to_text(index: int, message) -> str:
        participant = message.participant.name if message.participant is not None else "unknown"
        bits = "".join(str(int(b)) for b in message.plain_bits)
        decoded = "".join(str(int(b)) for b in message.decoded_bits)

        field_values = []
        for label in message.message_type:
            try:
                label_bits = "".join(
                    str(int(b)) for b in message.decoded_bits[label.start : label.end]
                )
                field_values.append("{}={}".format(label.name, label_bits))
            except Exception:
                pass

        lines = [
            "  Message {} (participant: {})".format(index, participant),
            "    bits:    {}".format(bits),
            "    decoded: {}".format(decoded),
            "    hex:     {}".format(message.plain_hex_str),
        ]
        if field_values:
            lines.append("    fields:  {}".format(", ".join(field_values)))

        return "\n".join(lines)
