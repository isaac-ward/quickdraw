Ideas of the user, do not incorporate into designs. Ignore. Just for the user to think about stuff

can we add another thing to the
  manfiold/ logs? its that horizon thing but as an animation. so it starts at h01, then interpolate to h02,
  h04, 8, 16, 32, 64 to show how accurate the denoising is over longer predictions. just in the dataspace over
  the position coordinates. hold on each timestep poitn cloud for 2 second, then tween (slow, fast, slow)
  between the next point cloud in 1 second. if a frmae is gonna eb identical don't rerender it. just shows how
  good the predictions are over long horizons. actually let me get some feedback. should this be in the
  eval_ood_horizon? and then we can keep doubling until we're quite far? we already would have generated the
  data right? thoughts? how many long horizon eval trajectories are there?