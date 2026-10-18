# Simulated policy update (re-indexing test)

`leave-policy-2026.pdf` here is version 2.0 of the leave policy. Compared to version 1.0 in
`../sample_docs/`, only these parts changed:

- the version line at the top (1 January -> revised 1 July 2026)
- the entitlement table: Casual Leave carry forward 4 -> 6 days
- section 3.1 Casual Leave: carry forward 4 -> 6 days

To simulate HR publishing the update, copy this file over the one in `../sample_docs/` and run
ingestion again. Expected result: only the chunks covering those three parts are re-embedded,
and the answer to "How many casual leaves can I carry forward?" changes from 4 to 6.
