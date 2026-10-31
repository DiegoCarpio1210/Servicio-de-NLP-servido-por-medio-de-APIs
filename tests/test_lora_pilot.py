import random
from experiments.train_lora_pilot import order_index
from experiments.prepare_lora_pilot import ch,qn

def test_resume_cursor_reconstructs_exact_sequence_without_global_rng_changes():
    random.seed(21);before=random.getstate()
    full=[order_index(i,17,42) for i in range(60)]
    assert full[:24]+[order_index(i,17,42) for i in range(24,60)]==full
    assert sorted(full[:17])==list(range(17))
    assert full[:17]!=full[17:34]
    assert random.getstate()==before

def test_context_and_question_audit_normalizes_common_format_variations():
    assert ch('Café  on\nEarth')==ch('CAFÉ on Earth')
    assert qn('What is this?')==qn('what is this')
