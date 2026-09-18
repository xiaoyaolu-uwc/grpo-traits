## Runs are degenerate because no abstentions whatsoever
### Problems & what I've tried
- Initial runs with a prompt which only notes the existence of unsolvable problems do not see any abstentions at all. Models output no thinking and direct answer.
- Tried adding CoT prompting, elicited thinking, but due to 64 token limit, thinking prompts were often punished since they were cut off.
- Tried hinting that models should consider whether the problem is solvable. Still no abstentions. On unsolvable problems, models assert "this is solvable" and go on to output an answer.
### Next steps:
Definitely try:
- Increase token limit to decrease punishment for thinking
- Increase temperature (currently 0.6), increase to 1.2
Potentially:
- Using strict reward grader to introduce variance on the answereable questions
- Add worked example of model abstaining (and directly answering) into the prompt
- Inject an abstention into initial rollouts