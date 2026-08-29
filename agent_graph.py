from __future__ import annotations


def report_writer(state):
  evidence = state["evidence"]

  prompt = """
  You are a financial research report writer.
  
  Use only the supplied evidence.
  Do not invent financial facts.
  Cite evidence sources.
  """
  pass