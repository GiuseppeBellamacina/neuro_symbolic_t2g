import logging


class BaseStreamer:
    # Stereamer has the functionality of updating PDA
    """
    Base class from which `.generate()` streamers should inherit.
    """

    def __init__(self, tokenizer, pda):
        self.tokenizer = tokenizer
        self.pda = pda
        self.is_first_call = True
        self.logit_processor = None  # Set by generate_grammar_parameters()

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        # Se il PDA ha già finito, esce subito
        if self.pda.eos() and self.is_first_call:
            logging.warning(
                "⚠️ PDA è già in stato finale (stack vuoto) PRIMA dell'inizio di una nuova generazione. "
                "Questo indica che manca un `pda.reset()` o che la grammatica è stata consumata interamente "
                "nella generazione precedente e il pda non è stato resettao -> stack vuoto []"
            )
            raise RuntimeError(
                "PDA already in final state before generation. "
                "Call pda.reset() before starting a new generation."
            )

        if self.pda.eos():
            self.is_first_call = True
            return

        # First call: consume all prompt tokens without processing
        if self.is_first_call:
            if hasattr(value, "tolist"):
                prompt_ids = (
                    value.tolist() if hasattr(value, "tolist") else [value.item()]
                )
            elif isinstance(value, (list, tuple)):
                prompt_ids = list(value)
            else:
                prompt_ids = [value]

            logging.debug(
                "First call: consuming %d prompt token(s) (first ID=%s, decoded=%r)",
                len(prompt_ids),
                prompt_ids[0] if prompt_ids else None,
                self.tokenizer.decode([prompt_ids[0]]) if prompt_ids else "",
            )
            self.is_first_call = False
            return

        # Da qui in poi arrivano i token generati uno per volta
        token_id = value.item() if hasattr(value, "item") else value
        logging.info(
            f"Token generato: {token_id} ({self.tokenizer.decode([token_id])})"
        )

        try:
            self.pda.next_state(token_id)
            logging.info(f"Stack PDA aggiornato: {self.pda.stack[::-1]}")
        except Exception as e:
            logging.error(f"Errore durante aggiornamento PDA con token {token_id}: {e}")
            logging.error(f"Stack corrente: {self.pda.stack}")
            raise

    def end(self):
        """
        Chiamato da .generate() per segnalare fine generazione.
        Resetta lo stato per permettere nuove generazioni.
        """
        logging.info("=== Fine generazione ===")

        # Verifica finale consistenza
        if not self.pda.eos():
            logging.warning(
                f"⚠ Generazione terminata ma stack PDA non vuoto: {self.pda.stack[::-1]}"
            )
        else:
            logging.info("✓ Stack PDA correttamente vuoto")

        # Reset completo dello stato per la prossima generazione
        # Reset per la prossima generazione
        self.is_first_call = True
        self.pda.reset()  # questo resetta anche il PDA
        if self.logit_processor is not None:
            self.logit_processor.reset()  # reset generation_ended flag

        logging.info("Streamer e PDA resettati per prossima generazione")
