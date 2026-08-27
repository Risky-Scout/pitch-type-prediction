from pitch_type_prediction.terminal_ui import table


def test_terminal_table_contains_headers_and_values():
    rendered = table(("Metric", "Value"), [("Log loss", "1.1793")], right_align={1})
    assert "Metric" in rendered
    assert "Log loss" in rendered
    assert "1.1793" in rendered
