"""Add policy identities to the upstream desktop UI, without changing battle rules.

Modified UI sources: desktop/.../ReplayFile.java, HandPanel.java, BattleWorkspace.java,
ReplayGameScreen.java @ 7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6
(https://github.com/voonhous/crforge). The original Apache 2.0 source headers remain.
"""

from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[1]
    / ".cache/replay-viewer/crforge/desktop/src/main/java/org/crforge/desktop"
)
MARKER = "CR RL policy identity labels (UI only)"


def patch(relative, edits):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError(f"Unexpected upstream source: {relative}: {old[:65]}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


patch(
    "replay/ReplayFile.java",
    [
        (
            "  private final ScenarioPlan plan;",
            "  private final ScenarioPlan plan;\n  // "
            + MARKER
            + "\n  private final List<String> playerLabels;",
        ),
        ("      ScenarioPlan plan) {", "      ScenarioPlan plan, List<String> playerLabels) {"),
        (
            "    this.plan = plan;",
            "    this.plan = plan;\n    this.playerLabels = List.copyOf(playerLabels);",
        ),
        (
            "        path, line, tables, ReplayCapture.of(document).orElse(null), "
            "header, refusals, plan);",
            "        path, line, tables, ReplayCapture.of(document).orElse(null), "
            "header, refusals, plan,\n"
            '        List.of(document.path("battle").path("avatar0").path("name").asText(""),\n'
            '                document.path("battle").path("avatar1").path("name").asText("")));',
        ),
        (
            "  /** Where the replay came from. */",
            "  public List<String> playerLabels() { return playerLabels; }\n\n"
            "  /** Where the replay came from. */",
        ),
    ],
)

patch(
    "render/HandPanel.java",
    [
        (
            "  private int side;",
            "  private int side;\n  // " + MARKER + '\n  private String playerLabel = "";',
        ),
        (
            "  void compact(boolean compact) {",
            '  void setPlayerLabel(String label) { playerLabel = label == null ? "" : label; }\n\n'
            "  void compact(boolean compact) {",
        ),
        (
            ': "SIDE "\n                + side\n                + " / "',
            ': (playerLabel.isBlank() ? "SIDE " + side : playerLabel + " / SIDE " + side)\n'
            '                + " / "',
        ),
    ],
)

patch(
    "render/BattleWorkspace.java",
    [
        (
            "  private final HandPanel bottomHand;",
            "  private final HandPanel bottomHand;\n  // "
            + MARKER
            + '\n  private List<String> playerLabels = List.of("", "");',
        ),
        (
            "  public void update(\n      BattleFrame frame,",
            "  public void setPlayerLabels(List<String> labels) { "
            "playerLabels = List.copyOf(labels); }\n\n"
            "  public void update(\n      BattleFrame frame,",
        ),
        (
            "    topHand.update(\n",
            "    topHand.setPlayerLabel(playerLabels.get("
            "view.getOrientation().sidesBottomFirst().get(1)));\n"
            "    bottomHand.setPlayerLabel(playerLabels.get(view.getOrientation().bottomSide()));\n"
            "    topHand.update(\n",
        ),
    ],
)

patch(
    "screen/ReplayGameScreen.java",
    [
        (
            "  private void showData() {",
            "  private void showData() {\n    // "
            + MARKER
            + '\n    workspace.setPlayerLabels(player.getReplay().playerLabels());\n'
            '    Gdx.graphics.setTitle("CRForge Replay | Side 0: " '
            '+ player.getReplay().playerLabels().get(0)\n'
            '        + " | Side 1: " + player.getReplay().playerLabels().get(1));',
        )
    ],
)
print("Policy labels patched in four UI files; battle core unchanged.")
