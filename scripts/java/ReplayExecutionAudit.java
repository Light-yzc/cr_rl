/*
 * Audits the actual outcomes of a saved crforge replay without a policy network.
 * API/dispatch reference: crforge 7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6,
 * core/.../replay/ReplayBattle.java and core/.../unit/Standard1v1Battle.java.
 * Upstream is Apache-2.0. Observation sentinels spend nothing and create no units.
 */
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.crforge.core.battle.Battle;
import org.crforge.core.battle.BattleCommand;
import org.crforge.core.battle.data.GameTables;
import org.crforge.core.battle.replay.ReplayBattle;
import org.crforge.core.battle.replay.ReplayScenario;
import org.crforge.core.battle.replay.ScenarioPlan;
import org.crforge.core.battle.unit.Standard1v1Battle;

public final class ReplayExecutionAudit {
  private static final ObjectMapper JSON = new ObjectMapper();

  private static void observe(Standard1v1Battle standard, int tick, Runnable observation) {
    standard.getBattle().queue(new BattleCommand() {
      public int tick() { return tick; }
      public void execute(Battle ignored) { observation.run(); }
    });
  }

  private static Map<String, Object> audit(GameTables tables, Path source) throws Exception {
    JsonNode raw = JSON.readTree(Files.readString(source));
    ScenarioPlan plan = new ReplayScenario(tables).translate(raw);
    ScenarioPlan setup = new ScenarioPlan(plan.seed(), plan.towers(), plan.decks(),
        plan.deckLevels(), plan.slotFlags(), plan.accounts(), plan.arenas(),
        plan.playerDataChoices(), List.of(), List.of(), plan.cardLevelMinimum());
    Standard1v1Battle standard = ReplayBattle.build(tables, setup);
    Map<Integer, ScenarioPlan.Play> plays = new HashMap<>();
    Map<Integer, ScenarioPlan.Ability> abilities = new HashMap<>();
    plan.plays().forEach(p -> plays.put(p.index(), p));
    plan.abilities().forEach(p -> abilities.put(p.index(), p));
    List<Map<String, Object>> commands = new ArrayList<>();
    for (int i = 0; i < plays.size() + abilities.size(); i++) {
      final int index = i;
      ScenarioPlan.Play play = plays.get(i);
      ScenarioPlan.Ability ability = abilities.get(i);
      int seat = play != null ? play.side() : ability.side();
      int given = play != null ? play.givenTick() : ability.givenTick();
      int run = play != null ? play.runTick() : ability.runTick();
      String name = "cmd" + i;
      Map<String, Object> row = new LinkedHashMap<>();
      row.put("index", i);
      row.put("name", name);
      row.put("side", seat);
      row.put("given_tick", given);
      row.put("run_tick", run);
      row.put("kind", play == null ? "ABILITY"
          : play.repeats() != null ? "MIRROR" : play.option() != null ? "VARIANT" : "CARD");
      row.put("card", play == null ? null : play.card());
      row.put("status", "queued_unrun");
      final int[] paidBefore = {0};
      final int[] elixirBefore = {0};
      observe(standard, run, () -> {
        var side = standard.getMatch().side(seat);
        paidBefore[0] = side.getSpent();
        elixirBefore[0] = side.getElixir();
        row.put("elixir_before", side.getElixir() / 10000.0);
        row.put("timeline_time_ms_at_command", standard.getMatch().getTimeline().getTimeMs());
        row.put("timeline_section_type", standard.getMatch().getTimeline().getSectionType());
        row.put("timeline_overtime", standard.getMatch().getTimeline().overtime());
        row.put("elixir_full_bar_ms", standard.getMatch().getTimeline().getFullBarMs());
      });
      // The same typed dispatch as ReplayBattle.build; only zero-effect observations differ.
      if (play == null) {
        standard.useAbility(given, run, seat, ability.objectId(), name);
      } else if (play.repeats() != null) {
        standard.playMirror(given, run, play.card(), play.level(), seat, play.x(), play.y(), name);
      } else if (play.option() != null) {
        standard.playVariant(given, run, play.card(), play.option().index(), play.level(),
            seat, play.x(), play.y(), name);
      } else {
        standard.play(given, run, standard.getWorld().getRecords().card(play.card()),
            play.level(), seat, play.x(), play.y(), name);
      }
      observe(standard, run, () -> {
        var side = standard.getMatch().side(seat);
        row.put("actual_run_tick", standard.getBattle().getTick());
        row.put("elixir_after", side.getElixir() / 10000.0);
        row.put("spent_elixir", (side.getSpent() - paidBefore[0]) / 10000.0);
        if (play == null) {
          var result = standard.getAbilityUses().stream()
              .filter(p -> p.name().equals(name)).findFirst().orElseThrow();
          row.put("refusal_code", result.outcome().code());
          row.put("status", result.outcome().code() == 0 ? "executed" : "refused");
          if (result.outcome().elixirBefore() != elixirBefore[0]
              || result.outcome().elixirAfter() != side.getElixir()) {
            throw new IllegalStateException("Ability command elixir observations disagree: " + index);
          }
        } else {
          var result = standard.getPlays().stream()
              .filter(p -> p.name().equals(name)).findFirst().orElseThrow();
          boolean placed = result.matchCode() == 0 && result.result() != null && result.result().placed();
          row.put("match_refusal_code", result.matchCode());
          row.put("placement_code", result.result() == null ? null : result.result().code());
          row.put("status", placed ? "executed" : "refused");
          Integer itemCost = result.mirror() != null ? result.mirror().cost()
              : result.variant() != null ? result.variant().cost()
              : result.evolution() != null ? result.evolution().cost() : null;
          row.put("item_cost", itemCost);
        }
        if (row.get("status").equals("refused") && ((Number) row.get("spent_elixir")).doubleValue() != 0) {
          throw new IllegalStateException("Refused command paid elixir: " + index);
        }
      });
      commands.add(row);
    }
    int expectedEnd = raw.path("endTick").asInt(-1);
    if (expectedEnd < 0) throw new IllegalArgumentException("Replay is missing endTick");
    Map<String, Integer> phaseTicks = new LinkedHashMap<>();
    Map<String, Integer> fullBarTicks = new LinkedHashMap<>();
    int overtimeTicks = 0;
    int activeTicks = 0;
    int checked = 0;
    while (standard.getBattle().getTick() < expectedEnd) {
      int before = standard.getBattle().getTick();
      standard.getBattle().step();
      if (before == standard.getBattle().getTick()) break;
      checked = ReplayBattle.checkItems(standard, plan, checked);
      var match = standard.getMatch();
      if (!match.isEnded()) {
        activeTicks++;
        phaseTicks.merge(String.valueOf(match.getTimeline().getSectionType()), 1, Integer::sum);
        fullBarTicks.merge(String.valueOf(match.getTimeline().getFullBarMs()), 1, Integer::sum);
        if (match.getTimeline().overtime()) overtimeTicks++;
      }
    }
    // An independent replay with no observations checks that instrumentation changed no outcome.
    Standard1v1Battle reference = ReplayBattle.build(tables, plan);
    int plainChecked = 0;
    while (reference.getBattle().getTick() < expectedEnd) {
      int before = reference.getBattle().getTick();
      reference.getBattle().step();
      if (before == reference.getBattle().getTick()) break;
      plainChecked = ReplayBattle.checkItems(reference, plan, plainChecked);
    }
    var match = standard.getMatch();
    var original = reference.getMatch();
    boolean same = reference.getBattle().getTick() == standard.getBattle().getTick()
        && original.getWinner() == match.getWinner() && original.crowns(0) == match.crowns(0)
        && original.crowns(1) == match.crowns(1);
    for (int side = 0; side < 2; side++) {
      same &= original.side(side).getSpent() == match.side(side).getSpent()
          && original.side(side).getElixir() == match.side(side).getElixir();
    }
    if (!same) throw new IllegalStateException("Audit instrumentation changed final replay outcome");
    Map<String, Object> result = new LinkedHashMap<>();
    result.put("source", source.toString());
    result.put("expected_end_tick", expectedEnd);
    result.put("actual_end_tick", standard.getBattle().getTick());
    result.put("duration_seconds", expectedEnd * 0.05);
    result.put("winner", match.getWinner());
    result.put("crowns", List.of(match.crowns(0), match.crowns(1)));
    result.put("native_outcome_parity", true);
    result.put("queued_commands", commands.size());
    result.put("commands", commands);
    result.put("active_simulation_seconds", activeTicks * 0.05);
    result.put("active_overtime_seconds", overtimeTicks * 0.05);
    result.put("active_ticks_by_section_type", phaseTicks);
    result.put("active_ticks_by_elixir_full_bar_ms", fullBarTicks);
    result.put("total_spent_elixir_by_side", List.of(match.side(0).getSpent() / 10000.0,
        match.side(1).getSpent() / 10000.0));
    return result;
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 3) throw new IllegalArgumentException("tables output.json replay.json ...");
    GameTables tables = GameTables.load(Path.of(args[0]));
    List<Map<String, Object>> results = new ArrayList<>();
    for (int i = 2; i < args.length; i++) results.add(audit(tables, Path.of(args[i])));
    JSON.writerWithDefaultPrettyPrinter().writeValue(Path.of(args[1]).toFile(), results);
  }
}
