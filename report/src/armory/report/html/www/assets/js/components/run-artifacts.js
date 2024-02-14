export default {
    props: {
        run: Object,
    },
    template: `
        <div class="flex gap-2 items-center my-2">
            <span>Chains</span>
            <select class="select select-bordered select-sm w-40">
                <option disabled>First chain?</option>
                <option selected>benign</option>
                <option>attacked</option>
                <option>defended</option>
            </select>
            <span>vs</span>
            <select class="select select-bordered select-sm w-40">
                <option disabled>Second chain?</option>
                <option>benign</option>
                <option>attacked</option>
                <option selected>defended</option>
            </select>
            <span class="border-l-2 pl-2">
                Batch
            </span>
            <select class="select select-bordered select-sm w-40">
                <option disabled>Which batch?</option>
                <option selected>5</option>
                <option>10</option>
            </select>
            <span class="border-l-2 pl-2">
                Sample
            </span>
            <select class="select select-bordered select-sm w-40">
                <option disabled>Which sample?</option>
                <option>1</option>
                <option selected>2</option>
            </select>
        </div>
        <div class="diff aspect-square mx-auto max-w-xl">
            <div class="diff-item-1">
                <img alt="attacked" src="./assets/img/batch_14_ex_5_pgd_no_feedback_defended.png" />
            </div>
            <div class="diff-item-2">
                <img alt="benign" src="./assets/img/batch_14_ex_5_benign_no_defense.png" />
            </div>
            <div class="diff-resizer"></div>
        </div>
    `,
};
