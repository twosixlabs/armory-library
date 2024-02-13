export default {
    props: {
        run: Object,
    },
    template: `
        <div class="gap-2 flex my-2">
            <select class="select select-bordered select-sm w-40">
                <option disabled>First chain?</option>
                <option selected>benign</option>
                <option>attacked</option>
                <option>defended</option>
            </select>
            vs
            <select class="select select-bordered select-sm w-40">
                <option disabled>Second chain?</option>
                <option>benign</option>
                <option>attacked</option>
                <option selected>defended</option>
            </select>
            <select class="select select-bordered select-sm w-40">
                <option disabled selected>Which batch?</option>
                <option>batch 5</option>
                <option>batch 10</option>
            </select>
            <select class="select select-bordered select-sm w-40">
                <option disabled selected>Which sample?</option>
                <option>sample 1</option>
                <option>sample 2</option>
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
