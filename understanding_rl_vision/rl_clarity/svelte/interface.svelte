<script>
  import Query from './query.svelte';
  import TrajectoryDisplay from './trajectory_display.svelte';
  import Navigator from './navigator.svelte';
  import Screen from './screen.svelte';
  import Graph from './graph.svelte';
  import AttributionViewer from './attribution_viewer.svelte';
  import Legend from './legend.svelte';
  import FeatureViewer from './feature_viewer.svelte';
  import { css_multiply } from './css_manipulate.js';

  export let input_layer = "input";
  export let layers = [];
  export let features = [];
  export let attributions = [];
  export let attribution_policy = false;
  export let attribution_single_channels = true;
  export let attribution_totals = [];
  export let colors = {
    features: [],
    actions: [],
    graphs: {
      v: "green",
      action: "red",
      action_group: "orange",
      advantage: "blue"
    },
    trajectory: "blue"
  };
  export let action_combos = [];
  export let action_groups = [];
  export let trajectories = {
    actions: [],
    policy_logits: [],
    values: [],
    advantages: []
  };
  export let bookmarks = {
    high: [],
    low: []
  };
  export let vis_defaults = {
    subdiv_mult: 0.5,
    expand_mult: 4
  };
  export let subdirs = {
    observations: "observations/",
    trajectories: "trajectories/",
    trajectories_scrub: "trajectories_scrub/",
    features: "features/",
    thumbnails: "thumbnails/",
    attribution: "attribution/",
    attribution_scrub: "attribution_scrub/",
    features_grids: "features_grids/",
    attribution_totals: "attribution_totals/"
  };
  export let formatting = {
    video_height: "0px",
    video_width: "0px",
    video_speed: 25,
    policy_display_height: "0px",
    policy_display_width: "0px",
    navigator_width: "0px",
    scrubber_height: "0px",
    scrubber_width: "0px",
    scrubber_visible_duration: 128,
    legend_item_height: "0px",
    legend_item_width: "0px",
    feature_viewer_height: "0px",
    feature_viewer_width: "0px",
    attribution_weight: 0.9
  };
  formatting.policy_display_width="40em";
  formatting.navigator_width="24em";
  export let init = {
    layer: layers[Math.floor(layers.length / 2)],
    attribution_kinds: [{type: "v", data: null}],
    attribution_residual: false
    /* attribution_options: [
     *   {direction: "non", channel: "all", show_trajectory: true},
     *   {direction: "pos", channel: "all", show_trajectory: true},
     *   {direction: "neg", channel: "all", show_trajectory: true}
     * ] */
  };
  export let json_preloaded = {};

  const action_htmls = action_combos.map(function(combo) {
    let right = false;
    let left = false;
    let up = false;
    let down = false;
    let non_arrows = []
    for (let button of combo) {
      if (button.toUpperCase() === "RIGHT") {
        right = true;
      }
      else if (button.toUpperCase() === "LEFT") {
        left = true;
      }
      else if (button.toUpperCase() === "UP") {
        up = true;
      }
      else if (button.toUpperCase() === "DOWN") {
        down = true;
      }
      else {
        non_arrows.push(button);
      }
    }
    let arrows = [];
    if (right && up) {
      arrows.push("&#8599;");
    }
    else if (left && up) {
      arrows.push("&#8598;");
    }
    else if (right && down) {
      arrows.push("&#8600;");
    }
    else if (left && down) {
      arrows.push("&#8601;");
    }
    else if (right) {
      arrows.push("&#8594;");
    }
    else if (left) {
      arrows.push("&#8592;");
    }
    else if (up) {
      arrows.push("&#8593;");
    }
    else if (down) {
      arrows.push("&#8595;");
    }
    if (arrows.length === 0 && non_arrows.length === 0) {
      return "no-op";
    }
    else {
      return arrows.concat(non_arrows).join("+");
    }
  });

  const num_trajectories = Math.max.apply(
    null, Object.values(trajectories).map(arr => arr.length));

  let video_state = {
    position: 0,
    velocity_direction: 0
  };

  let selected_attribution_id = {
    layer: init.layer,
    trajectory: 0
  };
  const get_selected_attribution_item = function(selected_attribution_id, attribution_items) {
    const default_item = {
      layer: null,
      trajectory: null,
    };
    for (let item of attribution_items) {
      if (item.layer === selected_attribution_id.layer &&
          item.trajectory === selected_attribution_id.trajectory) {
        return item;
      }
    }
    return default_item;
  };
  $: selected_attribution = get_selected_attribution_item(selected_attribution_id, attributions);
  $: selected_attribution_totals = get_selected_attribution_item(selected_attribution_id, attribution_totals);
  $: max_duration = (function() {
    return Math.max.apply(null, Object.values(trajectories).map(
      arr => arr[selected_attribution_id.trajectory].length));
  })();
  let attribution_residual = init.attribution_residual;

  let attribution_kinds = [];
  $: graphs = (function() {
    let trajectory = selected_attribution.trajectory;
    let graphs = [{
      type: "advantage",
      data: null,
      title: "advantage",
      series: trajectories.advantages[trajectory],
      dones: trajectories.dones[trajectory],
    }];
    for (let attribution_kind of attribution_kinds) {
      if (attribution_kind !== null) {
        let graph = {
          type: attribution_kind.type,
          data: attribution_kind.data
        };
        let duplicate = false;
        for (let existing_graph of graphs) {
          if (graph.type === existing_graph.type && graph.data === existing_graph.data) {
            duplicate = true;
          }
        }
        if (!duplicate) {
          if (graph.type === "v"){
            graph.title = "value function";
            graph.series = trajectories.values[trajectory];
          }
          else if (graph.type === "action") {
            graph.title = action_htmls[graph.data] + " logit";
            graph.series = trajectories.policy_logits[trajectory].map(
              logits => logits[graph.data]);
          }
          else if (graph.type === "action_group") {
            let actions = action_groups[graph.data];
            graph.title = "[ " + actions.map(action => action_htmls[action]).join(" | ") + " ] logit sum";
            graph.series = trajectories.policy_logits[trajectory].map(logits => logits.reduce(
              (total, logit, action) => total + (actions.indexOf(action) === -1 ? 0 : logit), 0));
          }
          else {
            graph = null;
          }
          if (graph !== null) {
            graph.dones = trajectories.dones[trajectory];
            graphs.push(graph);
          }
        }
      }
    }
    return graphs;
  })();
  let attribution_single_channel = null;
  $: {
    if (selected_attribution.layer === input_layer) {
      attribution_single_channel = null;
    }
  }

  let selected_feature_id = null;
  $: selected_feature = (function() {
    const default_feature = {
      layer: null,
      number: null,
      images: null,
      overlay_grids: null,
      metadata: null
    };
    if (selected_feature_id === null) {
      return default_feature;
    }
    for (let feature of features) {
      if (feature.layer === selected_feature_id.layer &&
          feature.number === selected_feature_id.number) {
        return feature;
      }
    }
    return default_feature;
  })();

  let show_feature_viewer = false;
</script>

<style>
 :global(.center-text) {
   display: -webkit-box;
   display: -webkit-flex;
   display: -moz-box;
   display: -ms-flexbox;
   display: flex;
   -webkit-flex-align: center;
   -ms-flex-align: center;
   -webkit-align-items: center;
   align-items: center;
   justify-content: center;
   text-align: center;
 }

 :global(.grayscale) {
   filter: gray;
   -webkit-filter: grayscale(1);
   filter: grayscale(1);
 }

 :global(.opaque-hover:hover) {
   filter: url("data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\'><filter id=\'opaque\'><feComponentTransfer><feFuncA type=\'table\' tableValues=\'1 1\'/></feComponentTransfer></filter></svg>#opaque");
 }

 :global(.striped) {
   background: repeating-linear-gradient(135deg, lightgray 0px, whitesmoke 10px, lightgray 20px);
 }

 :global(.underrule) {
   border-bottom: 1px solid gray;
 }

 :global(.pixelated){
   image-rendering: optimizeSpeed;
   image-rendering: -moz-crisp-edges;
   image-rendering: -o-crisp-edges;
   image-rendering: -webkit-optimize-contrast;
   image-rendering: optimize-contrast;
   image-rendering: crisp-edges;
   image-rendering: pixelated;
   -ms-interpolation-mode: nearest-neighbor;
 }

 .flex {
   display: -webkit-box;
   display: -moz-box;
   display: -webkit-flex;
   display: -ms-flexbox;
   display: flex;
 }

 .panel-slow {
   -webkit-box-flex: 0;
   -moz-box-flex: 0;
   -webkit-flex-grow: 0;
   -ms-flex: 0;
   flex-grow: 0;
   -webkit-flex-shrink: 0;
   -moz-flex-shrink: 0;
   -ms-flex: 0;
   flex-shrink: 0;
   z-index: 0;
   padding: 0em 0.5em;
 }

 .panel-fast {
   -webkit-box-flex: 1;
   -moz-box-flex: 1;
   -webkit-flex-grow: 1;
   -ms-flex: 1;
   flex-grow: 1;
   -webkit-flex-shrink: 1;
   -moz-flex-shrink: 1;
   -ms-flex: 1;
   flex-shrink: 1;
   z-index: 0;
   padding: 0em 0.5em;
 }

 .panel-not-last {
   border-right: 1px solid gray;
 }

 .trajectory-label {
   display: inline-block;
   white-space: nowrap;
   width: 100%;
   margin: 0.1em 0em;
   border: 1px solid gray;
   background-size: 100% 100%;
   line-height: 2em;
   font-weight: bold;
 }

 .layer-label {
   white-space: nowrap;
   padding: 0.5em;
   border: 1px solid gray;
   line-height: 3em;
 }

 .a {
   text-decoration: underline;
   color: #0000ee;
   cursor: pointer;
 }

 .indicator{
   margin: 0 auto;
   border-width: 0px 6px;
   border-color: #a3a3a3;
   border-style: solid;
   background: white;
 }
</style>

<svelte:window on:click={() => show_feature_viewer = false}/>

<Query
  bind:selected_attribution_id={selected_attribution_id}
  bind:video_state={video_state}
  max_duration={max_duration}
  num_trajectories={num_trajectories}
/>

<div class="flex">
  
  <div class="panel-slow panel-not-last">

    <h3 class="underrule">Trajectories</h3>
    <p>
      {#each Array.from(Array(num_trajectories).keys()) as trajectory}
        <label class="trajectory-label" style="background-image: url('{subdirs.trajectories_scrub + trajectory + '.png'}');">
          <input type="radio" bind:group={selected_attribution_id.trajectory} value={trajectory}>
          <span style="background-color: white;">{trajectory + 1}</span>
        </label>
        <br>
      {/each}
    </p>

    <h3 class="underrule">Bookmarks</h3>
    <p>Lowest advantage<br>episodes<br>(unexpected failures):</p>
    <p>
      {#each bookmarks.low as bookmark, bookmark_index}
        <span class="a" on:click={() => {selected_attribution_id.trajectory = bookmark[0]; video_state.position = bookmark[1];}}>
          trajectory {bookmark[0] + 1}, frame {bookmark[1] + 1}
        </span>
        <br>
      {/each}
    </p>
    <p>Highest advantage<br>episodes<br>(unexpected successes):</p>
    <p>
      {#each bookmarks.high as bookmark, bookmark_index}
        <span class="a" on:click={() => {selected_attribution_id.trajectory = bookmark[0]; video_state.position = bookmark[1];}}>
          trajectory {bookmark[0] + 1}, frame {bookmark[1] + 1}
        </span>
        <br>
      {/each}
    </p>

  </div>

  <div class="panel-slow panel-not-last">
    <h3 class="underrule">Layers</h3>
    <p style="text-align: center;">
      {#each layers as layer}
        <label class="layer-label"><input type="radio" bind:group={selected_attribution_id.layer} value={layer}> {layer}</label><br>
      {/each}
    </p>
  </div>

  <div class="panel-fast panel-not-last">

    <h3 class="underrule">Timeline</h3>

    <TrajectoryDisplay
      actions={trajectories.actions[selected_attribution.trajectory]}
      rewards={trajectories.rewards[selected_attribution.trajectory]}
      dones={trajectories.dones[selected_attribution.trajectory]}
      policy_logits={trajectories.policy_logits[selected_attribution.trajectory]}
      bind:state={video_state}
      action_htmls={action_htmls}
      action_colors={colors.actions}
      bold_color={colors.trajectory}
      policy_display_height={formatting.policy_display_height}
      policy_display_width={formatting.policy_display_width}
    />

    <Navigator
      bind:state={video_state}
      bind:speed={formatting.video_speed}
      width={formatting.navigator_width}
      max_duration={max_duration}
    />

    <div style="width: {formatting.scrubber_width};
                margin: 0 auto;
                background: whitesmoke;
                border: 1px solid gray;
                border-radius: 0.5em;
                box-shadow: inset 0 0 0.5em gray;">

      <div style="height: {css_multiply(formatting.scrubber_height, 0.2)};
                  width: 2px;
                  margin: 0 auto;
                  border-width: 0px 6px;
                  border-color: black;
                  border-style: solid;
                  opacity: 0.4;"
      ></div>

      <Screen
        image_dir={subdirs.trajectories_scrub}
        images={[selected_attribution.trajectory + ".png"]}
        durations={[max_duration]}
        bind:state={video_state}
        height={formatting.scrubber_height}
        width={formatting.scrubber_width}
        visible_duration={formatting.scrubber_visible_duration}
      />

      {#each graphs as graph}
        <Graph
          titles={[graph.title]}
          series={[graph.series]}
          dones={[graph.dones]}
          colors={[colors.graphs[graph.type]]}
          bind:state={video_state}
          height={formatting.scrubber_height}
          width={formatting.scrubber_width}
          visible_duration={formatting.scrubber_visible_duration}
        />
      {/each}

      <div class="indicator" style="height: {css_multiply(formatting.scrubber_height, 0.05)}; width: 2px;"></div>

      <div class="indicator" style="height: {css_multiply(formatting.scrubber_height, 0.05)}; width: 4px;"></div>

      <div class="indicator" style="height: {css_multiply(formatting.scrubber_height, 0.05)}; width: 8px;"></div>

      <div class="indicator" style="height: {css_multiply(formatting.scrubber_height, 0.05)}; width: 14px;"></div>

    </div>

    <div style="position: relative;
                padding: 0.5em;
                border: 1px solid black;
                border-radius: 0.5em;">
      <div style="position: absolute;
                  top: 0%;
                  left: 50%;
                  margin-top: -2px;
                  margin-left: -12px;
                  height: 4px;
                  width: 24px;
                  background-color: white;"
      ></div>
      <h3 class="underrule">{#if selected_attribution.layer !== input_layer}Attribution{:else}Gradients{/if}</h3>
      <AttributionViewer
        layer={selected_attribution.layer}
        trajectory={selected_attribution.trajectory}
        subdirs={subdirs}
        images={selected_attribution.images}
        metadata={selected_attribution.metadata}
        channel_totals_jsons={selected_attribution_totals.layer === null ? null : selected_attribution_totals.channels}
        residual_totals_jsons={selected_attribution_totals.layer === null ? null : selected_attribution_totals.residuals}
        totals_metadata={selected_attribution_totals.layer === null ? null : selected_attribution_totals.metadata}
        json_preloaded={json_preloaded}
        bind:state={video_state}
        bind:attribution_kinds={attribution_kinds}
        initial_attribution_kinds={init.attribution_kinds}
        action_htmls={action_htmls}
        action_groups={action_groups}
        max_duration={max_duration}
        video_height={formatting.video_height}
        video_width={formatting.video_width}
        attribution_weight={formatting.attribution_weight}
        attribution_or_gradients={selected_attribution.layer !== input_layer ? "attribution" : "gradients"}
        bind:attribution_residual={attribution_residual}
        attribution_policy={attribution_policy}
        attribution_single_channel={attribution_single_channel}
        channel_colors={colors.features}
      />
    </div>

  </div>

  <div class="panel-slow">

    {#if selected_attribution.layer !== input_layer}
      <h3 class="underrule">Attribution legend</h3>
      <p>
        Click to expand feature
        {#if attribution_single_channels}
          <br>Hover to isolate
        {/if}
      </p>
      <Legend
        image_dir={subdirs.thumbnails}
        image={selected_attribution.layer.replace(/\//g, "").replace(/_/g, "") + ".png"}
        colors={colors.features}
        item_height={formatting.legend_item_height}
        item_width={formatting.legend_item_width}
        show_residual={attribution_residual}
        bind:selected_channel={attribution_single_channel}
        enable_hover={attribution_single_channels}
        on:select={(event) => {selected_feature_id = { layer: selected_attribution.layer, number: event.detail }; show_feature_viewer = true;}}
      />
    {:else}
      <h3 class="underrule">Gradients legend</h3>
      <p>Colors correspond<br>to input colors</p>
    {/if}

    <h3 class="underrule">Hotkeys</h3>
    <p>
      <button>&#8592;</button> go backwards<br>
      <button>&#8594;</button> go forwards<br>
      <button>space</button> toggle play/pause<br>
    </p>
  
  </div>

</div>

<div style="display: {show_feature_viewer ? 'block' : 'none'};
            position: fixed;
            overflow: auto;
            z-index: 1;
            top: 2%;
            left: 50%;
            max-height: 85%;
            width: {formatting.feature_viewer_width};
            margin-left: {css_multiply(formatting.feature_viewer_width, -0.5/0.92)};
            padding: {css_multiply(formatting.feature_viewer_width, 0.04/0.92)};
            z-index: 1;
            background-color: white;
            border: 1px solid black;
            border-radius: 0.5em;"
     on:click={(event) => event.stopPropagation()}
>
  <FeatureViewer
    layer={selected_feature.layer}
    number={selected_feature.number}
    image_dir={subdirs.features}
    images={selected_feature.images}
    overlay_image_dir={subdirs.observations}
    overlay_image_grids_dir={subdirs.features_grids}
    overlay_image_grids_jsons={selected_feature.overlay_grids}
    metadata={selected_feature.metadata}
    metadata_initial_values={vis_defaults}
    json_preloaded={json_preloaded}
    height={formatting.feature_viewer_height}
    width={formatting.feature_viewer_width}
    on:close={() => show_feature_viewer = false}
  />
</div>
